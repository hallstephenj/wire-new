import os
import stripe
from fastapi import Request, HTTPException
from wire.db import get_conn

stripe.api_key = os.environ.get("STRIPE_SECRET_KEY")
PRO_PRICE_ID = os.environ.get("STRIPE_PRO_PRICE_ID")
WEBHOOK_SECRET = os.environ.get("STRIPE_WEBHOOK_SECRET")
APP_URL = os.environ.get("APP_URL", "http://localhost:8000")


async def create_checkout(user: dict) -> str:
    """Returns Stripe Checkout session URL."""
    session = stripe.checkout.Session.create(
        customer_email=user.get("email"),
        mode="subscription",
        line_items=[{"price": PRO_PRICE_ID, "quantity": 1}],
        success_url=f"{APP_URL}/settings?upgraded=1",
        cancel_url=f"{APP_URL}/settings",
        metadata={"user_id": user["id"]},
    )
    return session.url


async def create_portal(user: dict) -> str:
    """Returns Stripe Customer Portal URL."""
    session = stripe.billing_portal.Session.create(
        customer=user["stripe_customer_id"],
        return_url=f"{APP_URL}/settings",
    )
    return session.url


async def handle_webhook(request: Request):
    payload = await request.body()
    sig = request.headers.get("stripe-signature")
    try:
        event = stripe.Webhook.construct_event(payload, sig, WEBHOOK_SECRET)
    except Exception:
        raise HTTPException(400)

    conn = get_conn()
    if event["type"] == "checkout.session.completed":
        sess = event["data"]["object"]
        user_id = sess["metadata"]["user_id"]
        conn.execute("""
            UPDATE user_profiles SET subscription_status='pro',
            stripe_customer_id=?, stripe_subscription_id=?, updated_at=datetime('now')
            WHERE id=?
        """, (sess["customer"], sess["subscription"], user_id))

    elif event["type"] in ("customer.subscription.deleted", "customer.subscription.paused"):
        sub = event["data"]["object"]
        conn.execute("""
            UPDATE user_profiles SET subscription_status='cancelled', updated_at=datetime('now')
            WHERE stripe_subscription_id=?
        """, (sub["id"],))

    elif event["type"] == "customer.subscription.updated":
        sub = event["data"]["object"]
        status = "pro" if sub["status"] == "active" else "cancelled"
        conn.execute("""
            UPDATE user_profiles SET subscription_status=?, updated_at=datetime('now')
            WHERE stripe_subscription_id=?
        """, (status, sub["id"]))

    conn.commit()
    conn.close()
    return {"ok": True}

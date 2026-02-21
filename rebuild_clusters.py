"""Nuke all clusters, dedupe raw_items, and rebuild from scratch."""
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from wire.db import get_conn
from wire.cluster import assign_cluster

def rebuild():
    conn = get_conn()

    # Step 1: Deduplicate raw_items — keep one per (original_headline, source_name)
    print("Deduplicating raw_items...")
    dupes = conn.execute("""
        DELETE FROM raw_items WHERE rowid NOT IN (
            SELECT MIN(rowid) FROM raw_items GROUP BY original_headline, source_name
        )
    """)
    conn.commit()
    remaining = conn.execute("SELECT COUNT(*) FROM raw_items").fetchone()[0]
    print(f"  Removed duplicates, {remaining} unique items remain")

    # Step 2: Nuke all clusters and sources
    print("Clearing clusters and sources...")
    conn.execute("PRAGMA foreign_keys=OFF")
    conn.execute("DELETE FROM cluster_sources")
    conn.execute("DELETE FROM story_clusters")
    conn.execute("UPDATE raw_items SET cluster_id = NULL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.commit()

    # Step 3: Re-cluster all raw items ordered by published_at
    items = conn.execute("""
        SELECT id, original_headline, source_url, source_name, category, published_at
        FROM raw_items
        ORDER BY published_at ASC NULLS LAST
    """).fetchall()

    print(f"Re-clustering {len(items)} items...")
    for i, item in enumerate(items):
        cluster_id = assign_cluster(
            conn,
            item["original_headline"],
            item["source_url"],
            item["source_name"],
            item["category"],
            published_at=item["published_at"]
        )
        conn.execute("UPDATE raw_items SET cluster_id=? WHERE id=?", (cluster_id, item["id"]))

        if (i + 1) % 200 == 0:
            conn.commit()
            clusters = conn.execute("SELECT COUNT(*) FROM story_clusters").fetchone()[0]
            print(f"  {i+1}/{len(items)} items → {clusters} clusters")

    conn.commit()

    # Final stats
    total_clusters = conn.execute("SELECT COUNT(*) FROM story_clusters").fetchone()[0]
    by_cat = conn.execute("SELECT category, COUNT(*) FROM story_clusters GROUP BY category ORDER BY COUNT(*) DESC").fetchall()
    print(f"\nDone! {total_clusters} clusters from {len(items)} items")
    for cat, cnt in by_cat:
        print(f"  {cat}: {cnt}")

    conn.close()

if __name__ == "__main__":
    rebuild()

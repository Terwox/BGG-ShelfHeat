"""Quick test of the local BGG database."""
from shelfheat.bggdb import BGGLocalDB

db = BGGLocalDB()
print(db.stats())

print("\n--- Search: 'Agricola' ---")
for r in db.search_name("Agricola"):
    print(f"  {r['bgg_id']:>7}  {r['name']}")

print("\n--- Search: 'Shards of Infinity' ---")
for r in db.search_name("Shards of Infinity"):
    print(f"  {r['bgg_id']:>7}  {r['name']}")

print("\n--- Exact: 'Arcs' ---")
r = db.exact_name("Arcs")
print(f"  {r['bgg_id']:>7}  {r['name']}" if r else "  Not found")

print("\n--- Exact: 'Jarts' ---")
r = db.exact_name("Jarts")
print(f"  {r['bgg_id']:>7}  {r['name']}" if r else "  Not found (correct!)")

db.close()

from shelfheat.match import BGGCollection


def test_collection_loads_sample_bgg_csv():
    collection = BGGCollection.from_csv("test-collection.csv")

    assert len(collection.games) == 439
    assert collection.games[0] == {
        "name": "7 Wonders",
        "bgg_id": 68448,
        "play_count": 92,
        "last_played": "2021-11-29",
        "user_rating": 8.6,
    }

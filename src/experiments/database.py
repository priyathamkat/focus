import sqlite3


class BGVarDB:

    ANNOTATIONS_TABLE = "annotations"

    def __init__(self, path) -> None:
        self._connection = sqlite3.connect(path)
        self._cursor = self._connection.cursor()

        self._cursor.execute(
            f"CREATE TABLE if not exists {BGVarDB.ANNOTATIONS_TABLE} (file_name TEXT PRIMARY KEY, category TEXT, time TEXT, weather TEXT, locations TEXT, humans TEXT)",
        )

    def write_entry(self, file_name, category, time, weather, locations, humans):
        self._cursor.execute(
            f"REPLACE INTO {BGVarDB.ANNOTATIONS_TABLE} VALUES (?, ?, ?, ?, ?, ?)",
            (file_name, category, time, weather, ", ".join((sorted(locations))), humans),
        )
        self._connection.commit()

    def read_entries(self, categories=None, times=None, weathers=None, locations=None, humans=None):

        query = f"SELECT file_name, category, time, weather, locations FROM {BGVarDB.ANNOTATIONS_TABLE} WHERE "
        conditions = []
        if categories is not None:
            conditions.append(f"category IN {BGVarDB.stringify(categories)}")
        if times is not None:
            conditions.append(f"time IN {BGVarDB.stringify(times)}")
        if weathers is not None:
            conditions.append(f"weather IN {BGVarDB.stringify(weathers)}")
        if locations is not None:
            conditions.append(f"locations LIKE '%{', '.join((sorted(locations)))}%'")
        if humans is None:
            conditions.append(f"humans = 'no'")

        query += " AND ".join(conditions)
        results = self._cursor.execute(query)
        yield from results

    def __del__(self) -> None:
        self._connection.commit()
        self._connection.close()

    def populate_temp_table(self):
        # TODO: delete this later
        self.write_entry("cat/1.jpg", "cat", "day", "sunny", ["forest", "grass"])
        self.write_entry("cat/89.jpg", "cat", "day", "foggy", ["forest"])
        self.write_entry("dog/91.jpg", "dog", "day", "sunny", ["indoors"])
        self.write_entry("horse/19.jpg", "horse", "night", "none", ["grass", "water"])
        self.write_entry("car/23.jpg", "car", "day", "cloudy", ["snow", "street"])
        self.write_entry("plane/76.jpg", "plane", "day", "raining", ["street"])

    def check_if_table_exists(self, table_name):
        self._cursor.execute(
            f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'"
        )

        return self._cursor.fetchone()[0] != 0

    def clear_annotations(self):
        self._cursor.execute(f"DROP TABLE {BGVarDB.ANNOTATIONS_TABLE}")

    @staticmethod
    def stringify(values):
        if len(values) == 1:
            return f"('{values[0]}')"
        else:
            return str(tuple(values))


if __name__ == "__main__":
    temp_db = BGVarDB("./temp.db")
    import csv
    with open("/cmlscratch/pkattaki/void/bg-var/mturk-results/checked_redo_batch_results.csv") as f:
        reader = list(csv.DictReader(f, delimiter=","))
        for line in reader[:100]:
            try:
                temp_db.write_entry(line["image_name"], line["category"], line["time"], line["weather"], eval(line["locations"]))
            except sqlite3.IntegrityError as e:
                print(line["image_name"])
                raise e
    rows = temp_db.read_entries(categories=["car"])
    for row in rows:
        print(row)
    temp_db.clear_annotations()

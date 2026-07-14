from pathlib import Path
import duckdb


def write_store(db_path: Path, parquet_dir: Path, tables: dict) -> None:
    parquet_dir.mkdir(parents=True, exist_ok=True)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    try:
        for name, df in tables.items():
            con.register("tmp", df)
            con.execute(f"CREATE OR REPLACE TABLE {name} AS SELECT * FROM tmp")
            con.execute(f"COPY {name} TO '{parquet_dir / (name + '.parquet')}' (FORMAT PARQUET)")
            con.unregister("tmp")
    finally:
        con.close()

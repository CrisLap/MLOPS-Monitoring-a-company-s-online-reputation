from airflow import DAG
from airflow.operators.bash import BashOperator
from datetime import datetime

with DAG(
    "sentiment_pipeline",
    start_date=datetime(2024, 1, 1),
    schedule_interval="@weekly",
    catchup=False,
) as dag:

    train = BashOperator(
        task_id="train",
        bash_command="python training/train.py"
    )


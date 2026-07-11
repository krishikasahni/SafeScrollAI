import pandas as pd
from datetime import datetime

def save_history(filename, result, risk, confidence):

    os.makedirs("outputs", exist_ok=True)

    history_file = "outputs/history.csv"

    now = datetime.now()

    row = pd.DataFrame({

        "Date":[now.strftime("%d-%m-%Y")],

        "Time":[now.strftime("%H:%M:%S")],

        "Video":[filename],

        "Result":[result],

        "Risk":[risk],

        "Confidence (%)":[round(confidence,2)]

    })

    if os.path.exists(history_file):

        history = pd.read_csv(history_file)

        history = pd.concat(
            [history,row],
            ignore_index=True
        )

    else:

        history = row

    history.to_csv(
        history_file,
        index=False
    )
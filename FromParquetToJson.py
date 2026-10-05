import pandas as pd
import json

Data = pd.read_parquet('PendingRaw.parquet', engine='pyarrow')

Data["period"] = pd.to_datetime(Data["period"]).dt.strftime("%Y_%m")

Data = Data.rename( columns={"mo": "municipality_id" })

Data = Data.pivot_table(
    index=["period", "municipality_id"],
    columns="category_15",
    values="value",
    aggfunc="first"
).reset_index()

Data = Data.rename(columns={
    "Продовольствие": "food_share",
    "Здоровье": "medicine_share",
    "Транспорт": "transport_share",
    "Общественное питание": "restaurant_share",
    "Маркетплейсы": "services_share",
    "Все категории": "all_share"
})

DataDict = Data.to_dict(orient='records')

with open('Data.json', 'w', encoding='utf-8') as file:
    json.dump(DataDict, file, ensure_ascii = False, indent=4)
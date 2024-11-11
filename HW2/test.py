import pandas as pd
from xgboost import XGBRegressor

# 讀取資料
train = pd.read_csv('訓練資料0714train.csv')
test = pd.read_csv('測試資料其他欄位0728test.csv')

# 移除指定的欄位
columns_to_drop_1 = [f'Input_C_{i:03d}' for i in range(15, 39)]
columns_to_drop_2 = [f'Input_C_{i:03d}' for i in range(63, 83)]
columns_to_drop = columns_to_drop_1 + columns_to_drop_2 + ['Number']

train = train.drop(columns=columns_to_drop, errors='ignore')
test = test.drop(columns=columns_to_drop, errors='ignore')
# train.dropna(inplace=True)
train.fillna(train.mean(), inplace=True) # 使用平均值填補缺失值
test.fillna(test.mean(), inplace=True) # 使用平均值填補缺失值

# 定義目標欄位
target_columns = [
    'Input_A6_024', 'Input_A3_016', 'Input_C_013', 'Input_A2_016', 'Input_A3_017',
    'Input_C_050', 'Input_A6_001', 'Input_C_096', 'Input_A3_018', 'Input_A6_019',
    'Input_A1_020', 'Input_A6_011', 'Input_A3_015', 'Input_C_046', 'Input_C_049',
    'Input_A2_024', 'Input_C_058', 'Input_C_057', 'Input_A3_013', 'Input_A2_017'
]

# 分離特徵與目標變數
X_train = train.drop(columns=target_columns, errors='ignore')
y_train = train[target_columns]

# 訓練模型並進行預測
model = XGBRegressor(random_state=42)
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(test)

predictions_df = pd.DataFrame(y_pred, columns=target_columns)
results = pd.concat([predictions_df], axis=1)

# 將結果輸出為 CSV
results.to_excel('prediction_results.xlsx', index=False)
print("預測結果已儲存為 'prediction_results.xlsx'")
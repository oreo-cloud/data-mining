import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
import numpy as np

# 讀取資料
data = pd.read_csv('訓練資料0714train.csv')

# 移除指定的欄位
columns_to_drop_1 = [f'Input_C_{i:03d}' for i in range(15, 39)]
columns_to_drop_2 = [f'Input_C_{i:03d}' for i in range(63, 83)]
columns_to_drop = columns_to_drop_1 + columns_to_drop_2 + ['Number']

data = data.drop(columns=columns_to_drop, errors='ignore')
data.dropna(inplace=True)

# 定義目標欄位
target_columns = [
    'Input_A1_020', 'Input_A2_016', 'Input_A2_017', 'Input_A2_024', 
    'Input_A3_013', 'Input_A3_015', 'Input_A3_016', 'Input_A3_017', 
    'Input_A3_018', 'Input_A6_001', 'Input_A6_011', 'Input_A6_019', 
    'Input_A6_024', 'Input_C_013', 'Input_C_046', 'Input_C_049', 
    'Input_C_050', 'Input_C_057', 'Input_C_058', 'Input_C_096'
]

# 分離特徵與目標變數
X = data.drop(columns=target_columns, errors='ignore')
y = data[target_columns]


# 拆分資料集為 8:2 的訓練集和測試集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 訓練模型並進行預測
model = RandomForestRegressor(random_state=42)
model.fit(X_train, y_train)

# 預測測試集
y_pred = model.predict(X_test)

# 計算每個目標變數的 RMSE
rmse_scores = {}
for i, column in enumerate(target_columns):
    rmse = np.sqrt(mean_squared_error(y_test[column], y_pred[:, i]))
    rmse_scores[column] = rmse

# 顯示每個目標變數的 RMSE
for target, rmse in rmse_scores.items():
    print(f'RMSE for {target}: {rmse:.4f}')

# 將測試集的實際值和預測值合併為 DataFrame 並存為 CSV
y_test = y_test.reset_index(drop=True)
y_pred_df = pd.DataFrame(y_pred, columns=[f'Pred_{col}' for col in target_columns])
results = pd.concat([y_test, y_pred_df], axis=1)

# 將結果輸出為 CSV
results.to_csv('prediction_results.csv', index=False)
print("預測結果已儲存為 'prediction_results.csv'")

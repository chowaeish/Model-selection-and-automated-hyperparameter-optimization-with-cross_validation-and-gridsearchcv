import warnings
warnings.simplefilter(action='ignore', category=Warning)
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, VotingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import GridSearchCV, cross_val_score, cross_validate
from data_prep import *
from eda import *

df = pd.read_csv("hitters.csv")
check_df(df)


#######################################
# Feature Engineering
#######################################
# Tüm vuruşlara göre topa isabetli vurma oranı
df["succes_hits_rate"] = df["Hits"]/df["AtBat"]
# Tüm vuruşlara göre en değerli vuruş oranı
df["HmRun_rate"] = df["HmRun"]/df["AtBat"]
# Bir oyuncunun koşturduğu oyuncu sayısına göre yaptırdığı hata sayısı
df["Walks_according_to_RBI"] = df["Walks"]/df["RBI"]
# Oyuncunun kariyeri boyunca yaptığı isabetli vuruşun, tüm vuruşa oranı
df["succes_hits_rate_all_life"] = df["CHits"]/df["CAtBat"]
# Oyuncunun en değerli sayılarının tüm değerli sayılara oranı
df["CHmRun_according_to_HmRun"] = df["CHmRun"]/df["HmRun"]
# Oyuncunun yıllara göre ortalama attığı sayı
df["CRuns_according_to_Years"] = df["CRuns"]/df["Years"]
# 1986-1987 yılları arasında oyuncunun verimi = 3* sayı + 2* asist / 1* hata_sayısı
df["Efficiency"] = (3*df["Runs"] + 2*df["Assists"]) / df["Errors"]

#######################################
#Data Preprocessing
#######################################

cat_cols, num_cols, cat_but_car = grab_col_names(df)
for col in cat_cols:
    cat_summary(df, col)
# Label Encoding
for col in cat_cols:
    df.loc[:, col] = label_encoder(df, col)
# eksik değer
missing_values_table(df)
df["Salary"].fillna(df["Salary"].mean(),inplace=True)
df["CHmRun_according_to_HmRun"].fillna(df["CHmRun_according_to_HmRun"].mean(),inplace=True)
df["Walks_according_to_RBI"].fillna(df["Walks_according_to_RBI"].mean(),inplace=True)
missing_values_table(df)

# aykırı değer
for col in num_cols:
    print(col, check_outlier(df, col))

for col in num_cols:
    replace_with_thresholds(df, col)


#model
x = df.drop(["Salary"], axis=1)
y = df["Salary"]
######################################################
# Base Models
######################################################


models = [('LR', LinearRegression()),
          ("Ridge", Ridge()),
          ("Lasso", Lasso()),
          ("ElasticNet", ElasticNet()),
          ('KNN', KNeighborsRegressor()),
          ('CART', DecisionTreeRegressor()),
          ('RF', RandomForestRegressor()),
          ('SVR', SVR()),
          ('GBM', GradientBoostingRegressor()),
          ("XGBoost", XGBRegressor(objective='reg:squarederror')),
          ("LightGBM", LGBMRegressor()),
          ("CatBoost", CatBoostRegressor(verbose=False))]


for name, regressor in models:
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, x, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

######################################################
# Automated Hyperparameter Optimization
######################################################

        ### Önce Random ile başlangıc değerlerini bulalım ###

from sklearn.model_selection import RandomizedSearchCV

rf_model = RandomForestRegressor(random_state=17)

rf_random_params = {"max_depth": np.random.randint(5, 50, 5),
                    "max_features": [3, 5, 7, "auto", "sqrt"],
                    "min_samples_split": np.random.randint(2, 50, 20),
                    "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=20)]}


rf_random = RandomizedSearchCV(estimator=rf_model,
                               param_distributions=rf_random_params,
                               n_iter=200,  
                               cv=5,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)


rf_random.fit(x, y)

rf_random.best_params_

rf_random.best_score_


cart_model = DecisionTreeRegressor(random_state=17)
cart_params = {'max_depth': np.random.randint(5, 50, 5),
               "min_samples_split": np.random.randint(2, 50, 20)}

cart_random = RandomizedSearchCV(estimator=cart_model,
                               param_distributions=cart_params,
                               n_iter=200,
                               cv=5,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

cart_random.fit(x, y)

cart_random.best_params_

cart_random.best_score_


xgboost_model = XGBRegressor(random_state=17)

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": np.random.randint(5, 50, 5),
                  "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=20)],
                  "colsample_bytree": [0.3, 1]}

xgboost_random = RandomizedSearchCV(estimator=xgboost_model,
                               param_distributions=xgboost_params,
                               n_iter=200,
                               cv=5,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

xgboost_random.fit(x, y)

xgboost_random.best_params_

xgboost_random.best_score_

lgbm_model = LGBMRegressor(random_state=17)

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [int(x) for x in np.linspace(start=200, stop=1500, num=20)],
                   "colsample_bytree": [0.3, 1]}

lightgbm_random = RandomizedSearchCV(estimator=lgbm_model,
                               param_distributions=lightgbm_params,
                               n_iter=200,
                               cv=5,
                               verbose=True,
                               random_state=42,
                               n_jobs=-1)

lightgbm_random.fit(x, y)

lightgbm_random.best_params_

lightgbm_random.best_score_




#### GridSearch ile ######

cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

rf_params = {"max_depth": [5, 8, 15, None],
             "max_features": [5, 7, "auto"],
             "min_samples_split": [8, 15, 20],
             "n_estimators": [200, 500]}

xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200],
                  "colsample_bytree": [0.5, 0.8]}

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}

regressors = [("CART", DecisionTreeRegressor(), cart_params),
              ("RF", RandomForestRegressor(), rf_params),
              ('XGBoost', XGBRegressor(objective='reg:squarederror'), xgboost_params),
              ('LightGBM', LGBMRegressor(), lightgbm_params)]

best_models = {}

for name, regressor, params in regressors:
    print(f"########## {name} ##########")
    rmse = np.mean(np.sqrt(-cross_val_score(regressor, x, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE: {round(rmse, 4)} ({name}) ")

    gs_best = GridSearchCV(regressor, params, cv=5, n_jobs=-1, verbose=False).fit(x, y)

    final_model = regressor.set_params(**gs_best.best_params_)
    rmse = np.mean(np.sqrt(-cross_val_score(final_model, x, y, cv=10, scoring="neg_mean_squared_error")))
    print(f"RMSE (After): {round(rmse, 4)} ({name}) ")

    print(f"{name} best params: {gs_best.best_params_}", end="\n\n")

    best_models[name] = final_model

######################################################
# # Stacking & Ensemble Learning
######################################################

voting_reg = VotingRegressor(estimators=[('RF', best_models["RF"]),
                                         ('LightGBM', best_models["LightGBM"])])

voting_reg.fit(x, y)


np.mean(np.sqrt(-cross_val_score(voting_reg, x, y, cv=10, scoring="neg_mean_squared_error")))



########################################## ÖDEV 2 ##########################################################
def val_curve_params(model, x, y, param_name, param_range, scoring="roc_auc", cv=10):
    train_score, test_score = validation_curve(
        model, x=x, y=y, param_name=param_name, param_range=param_range, scoring=scoring, cv=cv)

    mean_train_score = np.mean(train_score, axis=1)
    mean_test_score = np.mean(test_score, axis=1)

    plt.plot(param_range, mean_train_score,
             label="Training Score", color='b')

    plt.plot(param_range, mean_test_score,
             label="Validation Score", color='g')

    plt.title(f"Validation Curve for {type(model).__name__}")
    plt.xlabel(f"Number of {param_name}")
    plt.ylabel(f"{scoring}")
    plt.tight_layout()
    plt.legend(loc='best')
    plt.show()

lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500],
                   "colsample_bytree": [0.7, 1]}





lgbm_model = LGBMClassifier(random_state=17)


for i in range(len(lightgbm_params)):
    val_curve_params(lgbm_model, x, y, lightgbm_params[i][0], lightgbm_params[i][1])






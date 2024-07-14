import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.optimize import minimize
from joblib import dump

df = pd.read_csv('dataset.csv')


df = pd.get_dummies(df, columns=['Material de Embalagem'], prefix='Material')


features = ['Altura', 'Largura', 'Profundidade', 'Peso do Produto',
            'Custo por Unidade de Material', 'Volume de Vendas',
            'Material_Papelão', 'Material_Plástico', 'Material_Isopor']
X = df[features]
y = df[['Altura', 'Largura', 'Profundidade']]  

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

modelo_rf = RandomForestRegressor(random_state=42)  
modelo_rf.fit(X_train, y_train) 



y_pred = modelo_rf.predict(X_test)

rmse = mean_squared_error(y_test, y_pred, squared=False)
print(f'RMSE: {rmse}')


parametros_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10],
}



grid_search = GridSearchCV(estimator=modelo_rf, param_grid=parametros_grid,
                           scoring='neg_mean_squared_error', cv=5, verbose=2)
grid_search.fit(X_train, y_train)

# 5. Obter o Melhor Modelo
melhor_modelo = grid_search.best_estimator_


y_pred = melhor_modelo.predict(X_test)

# 7. Avaliar o Modelo
print("Melhores Hiperparâmetros:", grid_search.best_params_)
print("-----------------------------------")
print(f'RMSE: {mean_squared_error(y_test, y_pred, squared=False)}')
print(f'MAE: {mean_absolute_error(y_test, y_pred)}')
print(f'R²: {r2_score(y_test, y_pred)}')


dump(melhor_modelo, 'logreg.joblib')

'''
def calcular_custo_embalagem(material, altura, largura, profundidade, custo_por_unidade):
    area = (2 * altura * largura) + (2 * altura * profundidade) + (2 * largura * profundidade)
    custo_total = area * custo_por_unidade
    return custo_total

def custo_embalagem(dimensoes, material_papelao, material_plastico, material_isopor, custo_material):
    altura, largura, profundidade = dimensoes

    print(f"Iteração: Dimensões: {altura:.2f}, {largura:.2f}, {profundidade:.2f}")

    # Determinar o material a partir das colunas codificadas
    if material_papelao:
        material = 'Papelão'
    elif material_plastico:
        material = 'Plástico'
    elif material_isopor:
        material = 'Isopor'
    else:
        material = 'Desconhecido'  # Tratar caso nenhum material seja selecionado

    # Calcular o custo do material com base nas dimensões e tipo de material
    custo_material_total = calcular_custo_embalagem(material, altura, largura, profundidade, custo_material)

    print(f"Custo Total: {custo_material_total:.2f}")
    return custo_material_total

def custo_embalagem(dimensoes, material_papelao, material_plastico, material_isopor, custo_material):
    altura, largura, profundidade = dimensoes

    print(f"Iteração: Dimensões: {altura:.2f}, {largura:.2f}, {profundidade:.2f}")

    # Definir o custo por unidade de área com base no material (ajuste os valores)
    if material_papelao:
        custo_por_unidade = 0.553400
        material = 'Papelão'
    elif material_plastico:
        custo_por_unidade = 0.518422
        material = 'Plástico'
    elif material_isopor:
        custo_por_unidade = 0.566766
        material = 'Isopor'
    else:
      custo_por_unidade = 0
      material = 'Desconhecido'


    # Calcular o custo do material usando o custo por unidade correto
    custo_material_total = calcular_custo_embalagem(
        material, altura, largura, profundidade, custo_por_unidade
    )

    print(f"Custo Total: {custo_material_total:.2f}")
    return custo_material_total


# 2. Função para Recomendar Dimensões com Otimização
def recomendar_dimensoes(produto, custo_material):
    # Obter previsões do modelo para cada dimensão
    dimensoes_iniciais = [25,25,25]
    dimensoes_previstas = melhor_modelo.predict(produto)[0]

    # Definir limites para as dimensões (restrições)
    limites = [(10, 100), (10, 100), (10, 100)]  # Exemplo: dimensões mínimas e máximas

    # Otimizar as dimensões usando a função de custo
    resultado_otimizacao = minimize(
        custo_embalagem,
        dimensoes_iniciais,

        args=(produto['Material_Papelão'].iloc[0],
              produto['Material_Plástico'].iloc[0],
              produto['Material_Isopor'].iloc[0],
              custo_material)

    )

    dimensoes_otimas = resultado_otimizacao.x
    return dimensoes_otimas
'''
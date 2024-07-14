import joblib
from scipy.optimize import minimize
import pandas as pd

def load_model():
    loaded_model = joblib.load(open('logreg.joblib', 'rb'))
    return loaded_model

def create_df( Altura,
               Largura,
               Profundidade,
               Peso,
               Custo,
               Volume,
               material):
    novo_produto = pd.DataFrame({
        'Altura':  [Altura],
        'Largura': [Largura],
        'Profundidade': [Profundidade],
        'Peso do Produto': [Peso],
        'Custo por Unidade de Material':  [Custo],
        'Volume de Vendas':  [Volume],
        'Material de Embalagem': [material] })
    return novo_produto

def transform(novo_produto, loaded_model):

    if novo_produto['Material de Embalagem'].iloc[0] == 'Papelão':
        novo_produto = pd.get_dummies(novo_produto, columns=['Material de Embalagem'], prefix='Material')
        novo_produto['Material_Plástico'] = 0
        novo_produto['Material_Isopor'] = 0
        novo_produto['Material_Papelão'] = 1

    elif novo_produto['Material de Embalagem'].iloc[0] == 'Plástico':
        novo_produto = pd.get_dummies(novo_produto, columns=['Material de Embalagem'], prefix='Material')
        novo_produto['Material_Papelão'] = 0
        novo_produto['Material_Isopor'] = 0
        novo_produto['Material_Plástico'] = 1

    elif novo_produto['Material de Embalagem'].iloc[0] == 'Isopor':
        novo_produto = pd.get_dummies(novo_produto, columns=['Material de Embalagem'], prefix='Material')
        novo_produto['Material_Papelão'] = 0
        novo_produto['Material_Plástico'] = 0
        novo_produto['Material_Isopor'] = 1
    
    
    colunas_treinamento = loaded_model.feature_names_in_


    novo_produto = novo_produto[colunas_treinamento]
    
    return novo_produto


def get_material_cost_per_unit(material):
    """
    Retorna o custo por unidade de área com base no material.
    """
    material_costs = {
        'Papelão': 0.553400,
        'Plástico': 0.518422,
        'Isopor': 0.566766
    }
    return material_costs.get(material, 0)

def calculate_packaging_cost(material, height, width, depth, cost_per_unit):
    """
    Calcula o custo total da embalagem com base nas dimensões e no tipo de material.
    """
    area = (2 * height * width) + (2 * height * depth) + (2 * width * depth)
    total_cost = area * cost_per_unit
    return total_cost

def get_selected_material(material_papelao, material_plastico, material_isopor):
    """
    Retorna o material selecionado com base nas colunas codificadas.
    """
    if material_papelao:
        return 'Papelão'
    elif material_plastico:
        return 'Plástico'
    elif material_isopor:
        return 'Isopor'
    else:
        return 'Desconhecido'

def calculate_total_packaging_cost(dimensoes, produto, custo_material):
    """
    Calcula o custo total da embalagem com base nas dimensões e no tipo de material.
    """
    altura, largura, profundidade = dimensoes
    material = get_selected_material(
        produto['Material_Papelão'].iloc[0],
        produto['Material_Plástico'].iloc[0],
        produto['Material_Isopor'].iloc[0]
    )
    custo_por_unidade = get_material_cost_per_unit(material)
    custo_material_total = calculate_packaging_cost(
        material, altura, largura, profundidade, custo_por_unidade
    )
    print(f"Iteração: Dimensões: {altura:.2f}, {largura:.2f}, {profundidade:.2f}")
    print(f"Custo Total: {custo_material_total:.2f}")
    return custo_material_total

def recommend_dimensions(produto, custo_material, loaded_model):
    """
    Recomenda as dimensões otimizadas da embalagem.
    """
    # Obter previsões do modelo para cada dimensão
    dimensoes_iniciais = [25, 25, 25]
    dimensoes_previstas = loaded_model.predict(produto)[0]

    # Definir limites para as dimensões (restrições)
    limites = [(dimensoes_previstas[0] * 0.8, dimensoes_previstas[0] * 1.2),
               (dimensoes_previstas[1] * 0.8, dimensoes_previstas[1] * 1.2),
               (dimensoes_previstas[2] * 0.8, dimensoes_previstas[2] * 1.2)]

    # Otimizar as dimensões usando a função de custo
    resultado_otimizacao = minimize(
        calculate_total_packaging_cost,
        dimensoes_iniciais,
        args=(produto, custo_material),
        bounds=limites
    )

    dimensoes_otimas = resultado_otimizacao.x
    return dimensoes_otimas




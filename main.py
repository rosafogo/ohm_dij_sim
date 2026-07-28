import os
import heapq  # Utilizado para a Fila de Prioridade no algoritmo de Dijkstra
import random
from time import sleep
from dataclasses import dataclass
from typing import List, Tuple

# =============================================================================
# Simulador simplificado de fluxo de corrente elétrica em matriz bidimensional
# =============================================================================
#                               ==============
#                                 OBSERVAÇÕES
#                               ==============
#
#       Esta aplicação é uma simulação discreta focada na demonstração de
#       algoritmos de busca em grafos (Dijkstra) aplicados à física de
#       circuitos simplificada (Lei de Ohm).
#
#       O modelo foca no cálculo de menor resistência em tempo real e não
#       pretende replicar efeitos complexos de eletromagnetismo contínuo.
#
#       No código original, a visualização deve ser feita pelo terminal.
#       A execução do código via IDE costuma dar problemas na identificação
#       da variável do terminal ($TERM) e impedir a execução corretamente.
#
# =============================================================================
# 1. DADOS (Estado)
# Estruturas que definem o modelo de domínio da simulação.
# =============================================================================

@dataclass
class Artifact:
    """
    Representa um componente físico no plano (um nó no grafo).
    A 'resistance' atua como o peso da aresta para o algoritmo de busca.
    """
    name: str
    resistance: float
    voltage: float = 0.0

@dataclass
class FlowResult:
    """
    Objeto de Transferência de Dados (DTO) que encapsula o resultado
    da tentativa de estabelecer uma corrente elétrica pelo plano.
    """
    path: List[Tuple[int, int]]  # Lista de coordenadas (linha, coluna) formando o circuito
    current: float  # Amperagem calculada da corrente
    message: str  # Status da operação (sucesso ou motivo da falha)

# =============================================================================
# 2. FÁBRICAS (Funções simples para criar componentes)
# Padrão Factory Method simplificado para instanciar componentes.
# =============================================================================

def create_battery_a() -> Artifact:
    # Polo positivo/gerador. Possui carga inicial de 120V.
    return Artifact("Battery_A", 64.0, 120.0)

def create_battery_b() -> Artifact:
    # Polo negativo/terra. Possui diferencial de potencial 0V.
    return Artifact("Battery_B", 64.0, 0.0)

def create_copper() -> Artifact:
    # Material condutor com baixíssima resistência (facilita o caminho).
    return Artifact("Copper", 0.001)

def create_empty() -> Artifact:
    # Representa o "ar" ou o vácuo. Resistência altíssima atua como um isolante,
    # forçando o algoritmo a evitar esse caminho.
    return Artifact("Empty", random.uniform(100000, 110000))

# =============================================================================
# 3. LÓGICA DE NEGÓCIO E FÍSICA (Comportamento)
# =============================================================================

def create_plane(width: int, height: int) -> List[List[Artifact]]:
    """Gera um grid puramente vazio (matriz 2D)."""
    return [[create_empty() for _ in range(width)] for _ in range(height)]

def add_artifact(grid: List[List[Artifact]], row: int, col: int, artifact: Artifact):
    """Insere um componente no plano físico, sobrescrevendo o espaço vazio."""
    grid[row][col] = artifact

def update_oscillations(grid: List[List[Artifact]]):
    """
    Oscila a resistência dos materiais dinâmicos do plano a cada frame.
    Simula fatores físicos do mundo real, como variação de temperatura ou
    imperfeições microscópicas no condutor que alteram a resistência em tempo real.
    """
    for row in grid:
        for artifact in row:
            if artifact.name == "Copper":
                artifact.resistance = round(random.uniform(0.0005, 0.005), 4)
            elif artifact.name == "Empty":
                artifact.resistance = random.uniform(100000, 110000)

def calculate_flow(grid: List[List[Artifact]]) -> FlowResult:
    """
    Núcleo da simulação: Calcula o caminho de menor resistência elétrica
    usando o Algoritmo de Dijkstra para encontrar a rota ideal da corrente.
    """
    height, width = len(grid), len(grid[0])
    start_node, end_node = None, None

    # Passo 1: Varredura inicial O(N*M) para encontrar os polos do circuito (Baterias)
    for i in range(height):
        for j in range(width):
            if grid[i][j].name == "Battery_A":
                start_node = (i, j)
            elif grid[i][j].name == "Battery_B":
                end_node = (i, j)

    # Fail-fast: Se o circuito não tiver as duas pontas, aborta o cálculo.
    if not start_node or not end_node:
        return FlowResult([], 0.0, "Erro: Bateria A ou B não encontradas.")

    # Passo 2: Configuração do algoritmo de Dijkstra
    # Dicionário de distâncias (resistência acumulada) inicializado com infinito
    distances = {(i, j): float('inf') for i in range(height) for j in range(width)}
    distances[start_node] = grid[start_node[0]][start_node[1]].resistance

    # Rastreia o caminho percorrido para reconstruir a rota no final
    came_from = {}

    # Fila de prioridade (Min-Heap) otimiza a busca para explorar sempre o nó
    # de menor resistência acumulada disponível no momento.
    pq = [(distances[start_node], start_node)]

    # Passo 3: Exploração dos caminhos
    while pq:
        current_dist, current_node = heapq.heappop(pq)

        # Otimização (Early exit): Para a busca assim que o destino é alcançado
        if current_node == end_node: break

        # Ignora nós obsoletos na fila de prioridade
        if current_dist > distances[current_node]: continue

        # Explora os 4 vizinhos adjacentes (Cima, Baixo, Esquerda, Direita) - Sem diagonais
        for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            ni, nj = current_node[0] + di, current_node[1] + dj

            # Verifica se o vizinho está dentro dos limites da matriz
            if 0 <= ni < height and 0 <= nj < width:
                neighbor = grid[ni][nj]

                # Regra de negócio: Impede que a energia flua livremente pelo "ar",
                # exigindo a existência de condutores.
                if neighbor.name == "Empty": continue

                # Relaxamento da aresta: Se este caminho for mais "fácil" (menor resistência), atualizamos
                new_dist = current_dist + neighbor.resistance
                if new_dist < distances[(ni, nj)]:
                    distances[(ni, nj)] = new_dist
                    came_from[(ni, nj)] = current_node
                    heapq.heappush(pq, (new_dist, (ni, nj)))

    # Trata caso de circuito desconectado (nenhum caminho possível encontrado)
    if distances[end_node] == float('inf'):
        return FlowResult([], 0.0, "Circuito aberto: Fluxo não estabelecido.")

    # Passo 4: Reconstrução do caminho (do fim para o começo)
    path = []
    curr = end_node
    while curr in came_from:
        path.append(curr)
        curr = came_from[curr]
    path.append(start_node)
    path.reverse()  # Inverte para ficar na ordem correta (Origem -> Destino)

    # Passo 5: Cálculo da Amperagem (Aplicação simplificada da Lei de Ohm: I = V/R)
    v_diff = abs(grid[start_node[0]][start_node[1]].voltage - grid[end_node[0]][end_node[1]].voltage)
    current = v_diff / distances[end_node] if distances[end_node] > 0 else 0

    return FlowResult(path, current, "Sucesso")

# =============================================================================
# 4. VISUALIZAÇÃO (Efeitos colaterais restritos ao terminal)
# Camada de apresentação separada estritamente da lógica de negócio.
# =============================================================================

def render_physical(grid: List[List[Artifact]]):
    """Exibe o layout estrutural do circuito e onde os componentes físicos estão."""
    print("--- PHYSICAL GRID (Artifacts) ---")
    for row in grid:
        print(" | ".join(f"{artifact.name:^10}" for artifact in row))
    print()

def render_wavegrid(grid: List[List[Artifact]], path: List[Tuple[int, int]]):
    """
    Renderiza a onda de energia dinamicamente baseada no caminho encontrado.
    Utiliza códigos ANSI de escape para imprimir cores no terminal.
    """
    YELLOW = '\033[93m'
    RESET = '\033[0m'
    print("--- WAVE GRID (Energyflow) ---")

    for i in range(len(grid)):
        row_str = []
        for j in range(len(grid[0])):
            if (i, j) in path:
                # Destaca visualmente a rota percorrida pela corrente elétrica*
                if (i, j) == path[0]:
                    name = "Source"
                elif (i, j) == path[-1]:
                    name = "End"
                else:
                    name = "Eletricity"
                row_str.append(f"{YELLOW}{name:^10}{RESET}")
            else:
                # Nós que não fazem parte do caminho ativo são mostrados como entropia/dissipados
                row_str.append(f"{'Entropy':^10}")
        print(" | ".join(row_str))
    print()

# =============================================================================
# --- LOOP PRINCIPAL (Motor da Simulação) ---
# =============================================================================

def main():
    # Inicializa o plano físico como uma matriz 5x5 vazia
    grid = create_plane(5, 5)

    # Posicionamento dos componentes no circuito
    add_artifact(grid, 0, 0, create_battery_a())
    add_artifact(grid, 4, 4, create_battery_b())

    # Desenha o caminho condutor (fio de cobre) de forma imperfeita
    # para forçar o algoritmo de Dijkstra a calcular as curvas
    copper_coords = [(0, 1), (1, 1), (1, 2), (1, 3), (2, 2), (3, 2), (3, 3), (3, 4), (1, 4), (2, 4)]
    for r, c in copper_coords:
        add_artifact(grid, r, c, create_copper())

    # Game Loop: Atualiza, calcula, limpa a Tela e desenha
    while True:
        # Comando multiplataforma para limpar o terminal a cada frame
        os.system('clear' if os.name == 'posix' else 'cls')

        # Atualiza o estado físico (oscilações)
        update_oscillations(grid)

        # Calcula a lógica (pathfinding)
        result = calculate_flow(grid)

        # Renderiza a UI
        render_physical(grid)
        render_wavegrid(grid, result.path)

        # Exibe os dados técnicos do fluxo atual
        if result.path:
            print(f"Corrente estabelecida: {result.current:.5f} A")
        else:
            print(result.message)

        # Controla o framerate em segundos
        sleep(0.05)

if __name__ == "__main__":
    main()
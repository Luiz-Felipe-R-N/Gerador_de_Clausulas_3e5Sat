import random
import time
import numpy as np
import matplotlib.pyplot as plt
from pysat.solvers import Glucose3
from scipy.interpolate import interp1d

n = 10
k = 3
alpha_inicio = 1.0
alpha_fim = 100.0
alpha_passo = 0.1
num_instancias = 50

info_experimento = f"n-{n}_k-{k}_alpha-{alpha_inicio}-{alpha_fim}".replace('.', '_')
nome_log = f"LOG_{info_experimento}.txt"
nome_grafico = f"GRAFICO_{info_experimento}.png"

def gerar_clausulas_validas(n_var, m_claus, k_lits):
    variaveis = list(range(1, n_var + 1))
    clausulas = set()
    while len(clausulas) < m_claus:
        vars_clausula = random.sample(variaveis, k_lits)
        clausula = [v * random.choice([1, -1]) for v in vars_clausula]
        if frozenset(clausula) not in clausulas:
            clausulas.add(frozenset(clausula))
    return [list(c) for c in clausulas]

def executar_testes_alpha(n_var, m_claus, k_lits, num_testes):
    resultados = {'satisfazivel': 0, 'tempo_total': 0.0, 'tempos': []}
    for _ in range(num_testes):
        clausulas = gerar_clausulas_validas(n_var, m_claus, k_lits)
        inicio = time.perf_counter()
        with Glucose3(bootstrap_with=clausulas) as solver:
            resultado = solver.solve()
        tempo = time.perf_counter() - inicio
        resultados['tempo_total'] += tempo
        resultados['tempos'].append(tempo)
        if resultado:
            resultados['satisfazivel'] += 1
    return resultados

def calcular_alpha_critico(alphas, probabilidades):
    try:
        f = interp1d(probabilidades, alphas)
        return float(f(0.5))
    except:
        return None

def registrar_log(nome_arquivo, dados, alpha_c):
    with open(nome_arquivo, 'w') as log:
        log.write("RELATÓRIO COMPLETO\n")
        log.write(f"Alpha crítico calculado: {alpha_c if alpha_c else 'N/A'}\n")
        log.write("="*50 + "\n")
        for alpha, info in dados.items():
            log.write(f"Alpha {alpha:.1f} (m={info['m_clausulas']}):\n")
            log.write(f"  Probabilidade SAT: {info['probabilidade']:.2%}\n")
            log.write(f"  Tempo total: {info['tempo_total']:.2f}s\n")
            log.write(f"  Tempo médio/instância: {info['tempo_medio_instancia']:.4f}s\n")
            log.write("-"*50 + "\n")

def gerar_graficos(dados, alpha_c):
    plt.figure(figsize=(14,6))
    
    plt.subplot(1,2,1)
    alphas = list(dados.keys())
    probabilidades = [d['probabilidade'] for d in dados.values()]
    plt.plot(alphas, probabilidades, 'k-', linewidth=2)
    if alpha_c:
        plt.axvline(alpha_c, color='r', linestyle='--', label=f'α_c = {alpha_c:.2f}')
    plt.xlabel('α (cláusulas/variáveis)')
    plt.ylabel('Probabilidade SAT')
    plt.grid(linestyle=':')
    plt.legend()
    
    plt.subplot(1,2,2)
    tempos = [d['tempo_medio_instancia'] for d in dados.values()]
    plt.plot(alphas, tempos, 'k-', linewidth=2)
    plt.xlabel('α (cláusulas/variáveis)')
    plt.ylabel('Tempo Médio (s)')
    plt.grid(linestyle=':')
    
    plt.tight_layout()
    plt.savefig(nome_grafico, dpi=300)
    plt.close()

dados_experimento = {}
valores_alpha = np.round(np.arange(alpha_inicio, alpha_fim + alpha_passo, alpha_passo), 1)

print(f"\nIniciando experimento {k}-SAT (n={n})")
for alpha in valores_alpha:
    m_clausulas = max(1, int(alpha * n))
    print(f"Processando alpha={alpha:.1f} (m={m_clausulas})... ", end='', flush=True)
    
    resultados = executar_testes_alpha(n, m_clausulas, k, num_instancias)
    
    dados_experimento[alpha] = {
        'm_clausulas': m_clausulas,
        'probabilidade': resultados['satisfazivel']/num_instancias,
        'tempo_total': resultados['tempo_total'],
        'tempo_medio_instancia': np.mean(resultados['tempos'])
    }
    print("Concluído")

alpha_c = calcular_alpha_critico(list(dados_experimento.keys()), 
                               [d['probabilidade'] for d in dados_experimento.values()])

registrar_log(nome_log, dados_experimento, alpha_c)
gerar_graficos(dados_experimento, alpha_c)

print("\nExperimento finalizado!")
print(f"Arquivos gerados: {nome_log}, {nome_grafico}")
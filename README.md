# CNN MNIST — Estudo Comparativo de Arquiteturas

> Projeto de análise experimental de Redes Neurais Convolucionais aplicadas ao dataset MNIST, com foco em avaliação sistemática do impacto de hiperparâmetros arquiteturais na performance de classificação.

---

## Sumário

1. [Visão Geral](#visão-geral)
2. [Dataset](#dataset)
3. [Ambiente e Dependências](#ambiente-e-dependências)
4. [Arquitetura Baseline](#arquitetura-baseline)
5. [Metodologia](#metodologia)
6. [Experimentos Realizados](#experimentos-realizados)
   - [Fase 1 — Baseline](#fase-1--baseline-3×3-maxpool)
   - [Fase 2A — Filtro 3×3 vs 5×5](#fase-2a--filtro-3×3-vs-5×5)
   - [Fase 2B — MaxPool vs AvgPool](#fase-2b--maxpool-vs-avgpool)
   - [Fase 2C — Profundidade: 2 vs 3 Camadas](#fase-2c--profundidade-2-vs-3-camadas-convolucionais)
   - [Fase 3 — Batch Size](#fase-3--impacto-do-batch-size)
   - [Fase 4 — Learning Rate](#fase-4--impacto-da-learning-rate)
   - [Fase 5 — Análise de Overfitting e Erros](#fase-5--análise-de-overfitting-e-erros)
7. [Tabela Comparativa Final](#tabela-comparativa-final)
8. [Conclusões](#conclusões)
9. [Referências](#referências)

---

## Visão Geral

Este projeto realiza um estudo experimental sistemático de Redes Neurais Convolucionais (CNNs) para classificação de dígitos manuscritos utilizando o dataset MNIST. A proposta central é conduzir **experimentos controlados** — alterando uma variável arquitetural por vez — para isolar e quantificar o impacto de cada decisão de design na performance final do modelo.

O projeto é estruturado como um exercício de **Data Science aplicado a Deep Learning**, com ênfase em:

- Definição de hipóteses antes de cada experimento
- Medição rigorosa de métricas (loss e accuracy em treino e validação)
- Análise crítica dos resultados com embasamento teórico
- Visualização de curvas de aprendizado, matrizes de confusão e erros

---

## Dataset

**MNIST** (Modified National Institute of Standards and Technology)

| Atributo | Valor |
|---|---|
| Imagens de treino | 60.000 |
| Imagens de teste | 10.000 |
| Dimensões | 28 × 28 pixels |
| Canais | 1 (escala de cinza) |
| Classes | 10 (dígitos de 0 a 9) |
| Formato dos tensores | `[N, 1, 28, 28]` |

O dataset é carregado via `torchvision.datasets.MNIST` com a transformação `ToTensor()`, que converte os valores de pixel de `[0, 255]` para `[0.0, 1.0]`.

---

## Ambiente e Dependências

```
Python        3.x
PyTorch       2.10.0+cu128
torchvision   compatível
scikit-learn  (confusion_matrix, ConfusionMatrixDisplay)
numpy
matplotlib
```

**Hardware utilizado:** GPU CUDA (Google Colab — `device: cuda`)

```python
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
torch.manual_seed(123)  # seed fixo para reprodutibilidade
```

---

## Arquitetura Baseline

A rede de referência (`Baseline3x3`) foi construída com filtros convolucionais 3×3 e MaxPooling, servindo como ponto de comparação para todos os experimentos.

```
Input:  [N, 1, 28, 28]
  ↓  Conv2d(1 → 32, kernel=3×3)  → ReLU → BatchNorm2d → MaxPool(2×2)
     Output: [N, 32, 13, 13]

  ↓  Conv2d(32 → 32, kernel=3×3) → ReLU → BatchNorm2d → MaxPool(2×2)
     Output: [N, 32, 5, 5]

  ↓  Flatten
     Output: [N, 800]           ← 32 × 5 × 5 = 800

  ↓  Linear(800 → 128) → ReLU → Dropout(p=0.2)
  ↓  Linear(128 → 128) → ReLU → Dropout(p=0.2)
  ↓  Linear(128 → 10)
     Output: [N, 10]            ← logits brutos
```

**Cálculo dos shapes convolucionais:**
```
output_size = (input - kernel + 1) / stride

Conv1:  (28 - 3 + 1) / 1 = 26×26
Pool1:  26 / 2           = 13×13
Conv2:  (13 - 3 + 1) / 1 = 11×11
Pool2:  11 / 2           = 5×5  (arredondado para baixo)
Flatten: 32 × 5 × 5      = 800
```

**Configuração de treinamento padrão:**

| Parâmetro | Valor |
|---|---|
| Função de perda | `nn.CrossEntropyLoss` |
| Otimizador | `Adam` (lr=0.001) |
| Batch size | 128 |
| Épocas | 5 |
| Dropout | p=0.2 |

---

## Metodologia

O projeto adota o princípio do **experimento controlado**: em cada fase, apenas uma variável é modificada em relação ao baseline, mantendo todas as demais constantes. Isso garante que diferenças observadas na performance sejam atribuíveis à variável em teste.

O ciclo de cada experimento segue a lógica de Data Science:

```
1. Hipótese      → o que esperamos que aconteça e por quê
2. Experimento   → treinamento com a variável modificada
3. Medição       → loss e accuracy em treino e validação por época
4. Análise       → comparação com o baseline + interpretação teórica
```

**Métricas coletadas em cada época:**
- `train_loss` — CrossEntropyLoss média sobre os batches de treino
- `val_loss` — CrossEntropyLoss média sobre o conjunto de teste
- `train_acc` — Accuracy média sobre os batches de treino
- `val_acc` — Accuracy média sobre o conjunto de teste

**Gap de overfitting** é definido como `train_acc - val_acc`. Valores positivos e crescentes indicam que a rede está memorizando em vez de generalizar.

---

## Experimentos Realizados

### Fase 1 — Baseline (3×3, MaxPool)

Treinamento da arquitetura de referência por 5 épocas.

| Época | Loss Treino | Acc Treino | Loss Val | Acc Val |
|---|---|---|---|---|
| 1 | 0.1649 | 95.20% | 0.0460 | 98.59% |
| 2 | 0.0488 | 98.50% | 0.0316 | 98.98% |
| 3 | 0.0343 | 98.95% | 0.0258 | 99.15% |
| 4 | 0.0272 | 99.13% | 0.0384 | 98.86% |
| **5** | **0.0226** | **99.31%** | **0.0296** | **99.10%** |

**Observação:** A loss de validação na época 4 sofreu um pequeno aumento em relação à época 3 (0.0384 vs 0.0258), evidenciando oscilação típica do processo de otimização com Adam em 5 épocas. Comportamento esperado e sem sinal de overfitting relevante.

---

### Fase 2A — Filtro 3×3 vs 5×5

**Hipótese:** Filtros maiores (5×5) capturam padrões de maior escala, mas perdem resolução espacial mais rapidamente, podendo prejudicar a extração de features em imagens pequenas.

**Arquitetura `Filtro5x5`:** idêntica ao baseline, substituindo todos os `kernel_size=(3,3)` por `(5,5)`. O shape resultante exige ajuste na primeira camada linear:

```
Conv1: (28 - 5 + 1) = 24×24  →  Pool: 12×12
Conv2: (12 - 5 + 1) = 8×8    →  Pool: 4×4
Flatten: 32 × 4 × 4 = 512    ← menos features que o 3×3 (800)
```

| Época | Loss Treino | Acc Treino | Loss Val | Acc Val |
|---|---|---|---|---|
| 1 | 0.1475 | 95.74% | 0.0345 | 98.84% |
| 2 | 0.0403 | 98.81% | 0.0278 | 99.17% |
| 3 | 0.0287 | 99.14% | 0.0314 | 99.03% |
| 4 | 0.0215 | 99.32% | 0.0274 | 99.22% |
| **5** | **0.0179** | **99.43%** | **0.0258** | **99.26%** |

**Resultado:** O filtro 5×5 apresentou accuracy de validação final de **99.26%** contra **99.10%** do baseline 3×3 — uma diferença de +0.16 p.p. a favor do 5×5 neste experimento específico. No entanto, a loss de validação do 5×5 (0.0258) foi menor que a do 3×3 (0.0296), sugerindo melhor calibração probabilística.

**Análise crítica:** O MNIST é um dataset de alta regularidade — fundo preto uniforme, dígitos centralizados e traços consistentes. Nesse cenário, o campo receptivo maior do filtro 5×5 consegue capturar a estrutura global dos dígitos de forma eficiente já nas primeiras camadas, compensando a perda de resolução espacial. Em datasets mais complexos (texturas finas, objetos em escala variável), a vantagem do 3×3 se tornaria mais pronunciada, conforme demonstrado pela VGGNet (Simonyan & Zisserman, 2014).

---

### Fase 2B — MaxPool vs AvgPool

**Hipótese:** O AveragePooling, ao calcular a média da região, preserva informação difusa e pode ter melhor generalização. O MaxPooling, ao selecionar o valor máximo, é mais agressivo na seleção de features marcantes.

**Arquitetura `BaselineAvgPool`:** idêntica ao baseline, substituindo `nn.MaxPool2d` por `nn.AvgPool2d`.

| Época | Loss Treino | Acc Treino | Loss Val | Acc Val |
|---|---|---|---|---|
| 1 | 0.1626 | 95.32% | 0.0415 | 98.62% |
| 2 | 0.0476 | 98.54% | 0.0381 | 98.82% |
| 3 | 0.0352 | 98.93% | 0.0293 | 99.11% |
| 4 | 0.0284 | 99.15% | 0.0321 | 98.97% |
| **5** | **0.0245** | **99.26%** | **0.0203** | **99.35%** |

**Resultado:** O AvgPool obteve a **melhor accuracy de validação entre todos os modelos** com 5 épocas: **99.35%**, além da menor loss de validação: **0.0203**.

**Análise crítica:** O resultado surpreende pela margem. Uma hipótese explicativa é que o AvgPool, ao suavizar os feature maps, atua como um regularizador implícito — reduzindo a sensibilidade a pixels ruidosos e aumentando a generalização. Este efeito regularizador do AvgPool em datasets simples é discutido por Boureau, Ponce & LeCun (2010) em seu estudo teórico sobre operações de pooling.

---

### Fase 2C — Profundidade: 2 vs 3 Camadas Convolucionais

**Hipótese:** Uma camada convolucional adicional permite aprender representações hierárquicas mais ricas, potencialmente melhorando a accuracy.

**Arquitetura `TresCamadas`:**

```
Conv1: 1  → 32  (3×3)  → BN → ReLU → MaxPool   → 13×13
Conv2: 32 → 64  (3×3)  → BN → ReLU → MaxPool   → 5×5
Conv3: 64 → 64  (3×3)  → BN → ReLU             → 3×3
Flatten: 64 × 3 × 3 = 576
Linear(576 → 128) → ReLU → Dropout(p=0.3)
Linear(128 → 10)
```

| Época | Loss Treino | Acc Treino | Loss Val | Acc Val |
|---|---|---|---|---|
| 1 | 0.1222 | 96.57% | 0.0363 | 98.84% |
| 2 | 0.0390 | 98.84% | 0.0340 | 99.00% |
| 3 | 0.0274 | 99.11% | 0.0346 | 98.84% |
| 4 | 0.0205 | 99.35% | 0.0272 | 99.14% |
| **5** | **0.0158** | **99.51%** | **0.0433** | **98.87%** |

**Resultado:** A rede de 3 camadas apresentou o **maior overfitting** do experimento: na época 5, a accuracy de treino (99.51%) supera a de validação (98.87%) em **0.64 p.p.**, enquanto a loss de validação (0.0433) é a mais alta de todos os modelos. A rede aprendeu mais rápido (melhor acc no treino), mas generalizou pior.

**Análise crítica:** Para imagens 28×28 com apenas 10 classes, uma terceira camada convolucional adiciona capacidade expressiva desnecessária. Com mais parâmetros e complexidade sem regularização proporcional (seria necessário aumentar o dropout ou usar weight decay), a rede começa a memorizar padrões específicos do treino. Este é um exemplo clássico de overfitting por excesso de capacidade do modelo, tema central discutido em Goodfellow, Bengio & Courville (2016), capítulo 5.

---

### Fase 3 — Impacto do Batch Size

**Modelos testados:** Baseline 3×3 com batch sizes de 32, 128 e 1024.

| Batch Size | Acc Val (época 5) | Loss Val (época 5) | Observação |
|---|---|---|---|
| **32** | **99.20%** | 0.0274 | Convergência suave, mais ruidosa |
| 128 | 99.15% | 0.0304 | Comportamento padrão |
| 1024 | 99.03% | 0.0277 | Época 1 muito lenta (acc: 97.23%) |

**Análise crítica:**

O batch size 1024 arrancou de forma significativamente pior na época 1 (97.23% vs ~98.6% dos demais), demonstrando o impacto de batches muito grandes na qualidade dos gradientes iniciais. Batches grandes calculam gradientes mais "estáveis" (menos ruidosos), mas esse ruído do batch pequeno é, paradoxalmente, benéfico — ele funciona como regularizador implícito e ajuda a escapar de mínimos locais ruins. Este fenômeno é formalmente discutido por Keskar et al. (2017) em *"On Large-Batch Training for Deep Learning"*, que demonstra que batches grandes tendem a convergir para mínimos mais "aguçados" (sharp minima) com menor generalização.

---

### Fase 4 — Impacto da Learning Rate

**Learning rates testadas:** 0.1 (alto), 0.001 (padrão Adam), 0.00001 (baixo).

| LR | Loss Treino (ép. 5) | Loss Val (ép. 5) | Comportamento |
|---|---|---|---|
| 0.1 | 2.3070 | 2.3026 | **Divergência total** — loss estagnada em ~2.3 (equivale a chute aleatório em 10 classes) |
| **0.001** | **0.0224** | **0.0265** | **Convergência ideal** |
| 0.00001 | 0.3193 | 0.1910 | Aprendizado muito lento — ainda longe de convergir em 5 épocas |

**Análise crítica:**

A learning rate 0.1 resultou em divergência imediata. O Adam com LR tão alto realiza passos tão grandes nos pesos que o modelo salta continuamente sobre mínimos da função de perda sem convergir — a loss ficou presa em ~2.3, que é exatamente `-log(1/10)`, o valor esperado para classificação aleatória em 10 classes.

A LR 0.00001 aprendeu, mas muito devagar. Em 5 épocas a loss de validação ainda estava em 0.1910 — longe dos 0.0265 do modelo com LR padrão. Isso demonstra que LR baixa demais não compromete a capacidade de aprender, mas exige muito mais épocas para convergir.

O valor padrão do Adam (lr=0.001) é amplamente reconhecido como um bom ponto de partida para a maioria das tarefas, conforme demonstrado em Kingma & Ba (2015), o paper original do algoritmo Adam.

---

### Fase 5 — Análise de Overfitting e Erros

**Baseline treinado por 10 épocas** para observar o comportamento de longo prazo.

| Época | Loss Treino | Acc Treino | Loss Val | Acc Val |
|---|---|---|---|---|
| 1 | 0.1649 | 95.17% | 0.0468 | 98.48% |
| 5 | 0.0226 | 99.28% | 0.0237 | 99.29% |
| 8 | 0.0132 | 99.57% | 0.0287 | 99.26% |
| 10 | 0.0148 | 99.52% | 0.0379 | 99.06% |

A partir da época 5, a loss de treino continua caindo (0.0226 → 0.0132) enquanto a loss de validação começa a oscilar e subir levemente (0.0237 → 0.0379). Este é o sinal clássico de início de overfitting — a rede começa a memorizar características específicas do conjunto de treino.

**Matriz de Confusão — Top 3 confusões:**

| Real | Previsto | Erros |
|---|---|---|
| **3** | **5** | 10 |
| **9** | **4** | 9 |
| **7** | **2** | 6 |

Essas confusões são visualmente interpretáveis: o `3` e o `5` compartilham curvas similares na parte superior; o `9` e o `4` têm traços verticais e fechamentos parecidos; o `7` e o `2` podem ser confundidos em caligrafia cursiva.

---

## Tabela Comparativa Final

| Modelo | Acc Treino | Acc Val | Loss Val | Gap Overfitting |
|---|---|---|---|---|
| Baseline 3×3 (MaxPool) | 99.31% | 99.10% | 0.0296 | +0.21% |
| Filtro 5×5 | 99.43% | 99.26% | 0.0258 | +0.17% |
| **AvgPool** | 99.26% | **99.35%** | **0.0203** | **−0.09%** |
| 3 Camadas Conv | **99.51%** | 98.87% | 0.0433 | +0.64% |

> **Melhor acc de validação:** AvgPool (99.35%)  
> **Menor loss de validação:** AvgPool (0.0203)  
> **Menor overfitting:** AvgPool (gap negativo — val > train)  
> **Maior overfitting:** 3 Camadas Conv (+0.64%)

---

## Conclusões

### 1. AvgPool superou MaxPool neste contexto
O AvgPool atingiu a melhor accuracy de validação (99.35%) e a menor loss (0.0203), com gap de overfitting negativo. A suavização introduzida pela média atua como regularizador implícito, benéfico para um dataset de baixa variância como o MNIST.

### 2. Filtros 5×5 performaram bem em imagens pequenas
Contrariando a intuição baseada em VGGNet, o filtro 5×5 teve desempenho ligeiramente superior ao 3×3 neste dataset específico. Em imagens 28×28 com padrões simples, o campo receptivo maior consegue capturar a estrutura dos dígitos de forma eficiente já na primeira camada. A vantagem do 3×3 se tornaria mais evidente em imagens maiores e mais complexas.

### 3. Profundidade exige regularização proporcional
A rede com 3 camadas convolucionais apresentou o maior overfitting (gap de 0.64%). Adicionar capacidade ao modelo sem aumentar a regularização (dropout maior, weight decay, data augmentation) resulta em memorização em vez de generalização.

### 4. Learning rate é o hiperparâmetro mais crítico
A diferença entre lr=0.1 (divergência total, loss~2.3) e lr=0.001 (convergência ótima, loss~0.026) é de 100x. Nenhuma escolha arquitetural tem impacto tão dramático na performance quanto uma learning rate mal calibrada.

### 5. Batch size impacta velocidade de convergência
Batch size 1024 apresentou performance significativamente inferior na época 1 (97.23%), embora convergisse próximo dos demais modelos ao final. Batch sizes menores introduzem ruído nos gradientes que, paradoxalmente, ajuda a escapar de mínimos locais e melhora a generalização.

### 6. O ponto ótimo de parada está entre as épocas 3 e 5
A análise de 10 épocas mostrou que o modelo começa a apresentar overfitting a partir da época 6, com a loss de validação oscilando e subindo enquanto a de treino continua caindo. Em produção, seria recomendável implementar Early Stopping monitorando a loss de validação.

---

## Referências

- **LeCun, Y. et al. (1998).** Gradient-Based Learning Applied to Document Recognition. *Proceedings of the IEEE*, 86(11), 2278–2324.

- **Simonyan, K., & Zisserman, A. (2014).** Very Deep Convolutional Networks for Large-Scale Image Recognition. *arXiv:1409.1556*. — Referência central para a superioridade de filtros 3×3 empilhados.

- **Goodfellow, I., Bengio, Y., & Courville, A. (2016).** *Deep Learning*. MIT Press. — Capítulos 9 (CNNs) e 7 (Regularização).

- **Kingma, D. P., & Ba, J. (2015).** Adam: A Method for Stochastic Optimization. *arXiv:1412.6980*. — Paper original do otimizador Adam, justificando lr=0.001 como padrão.

- **Keskar, N. S. et al. (2017).** On Large-Batch Training for Deep Learning: Generalization Gap and Sharp Minima. *arXiv:1609.04836*. — Fundamenta o impacto do batch size na generalização.

- **Boureau, Y. L., Ponce, J., & LeCun, Y. (2010).** A Theoretical Analysis of Feature Pooling in Visual Recognition. *ICML 2010*. — Análise teórica comparando MaxPool e AvgPool.

---

*Projeto desenvolvido como parte de um estudo estruturado de Redes Neurais Convolucionais, com foco em ciência de dados aplicada e análise experimental de hiperparâmetros.*

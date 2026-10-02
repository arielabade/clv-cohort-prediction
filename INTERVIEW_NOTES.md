# Notas para entrevista — clv-cohort-prediction

Documento interno. Não faz parte da documentação pública do projeto.

---

## As 3 decisões técnicas mais importantes

### 1. Dois splits, não um

**O que fiz:** split temporal (calibração até 2011-06-09, holdout nos 183 dias seguintes) **e** split
por cliente (3.478 treino / 1.491 teste).

**Por quê o temporal:** split aleatório sobre transações deixaria compras futuras de um cliente
treinarem um modelo que depois é avaliado nas compras anteriores dele. É vazamento do futuro para o
passado, e infla toda métrica.

**Por quê o segundo:** o BG/NBD não usa rótulo — ele poderia ser ajustado em todos os clientes. Mas
aí ele seria avaliado em clientes que viu, enquanto o LightGBM seria avaliado em clientes que não viu.
A comparação não seria sobre qual modelo é melhor, seria sobre quem teve mais informação. Ajusto os
dois nos mesmos 3.478 e avalio nos mesmos 1.491.

### 2. Colapsar invoices do mesmo dia em uma ocasião de compra

**O que fiz:** agrego por cliente-dia antes de calcular frequência.

**Por quê:** o BG/NBD assume no máximo uma transação por unidade de tempo, e a unidade aqui é o dia.
No dataset, 9,1% dos dias-cliente têm mais de uma nota — é uma sessão de compra fracionada em vários
documentos, não duas decisões de compra. Contar por invoice inflaria a frequência observada em 11,7%
e, com ela, toda previsão de número de compras.

**Como descobri:** olhando as primeiras linhas, o cliente 12346 tinha três invoices no mesmo dia.
Medi o impacto antes de decidir, em vez de assumir que era irrelevante.

### 3. Avaliar por decil, não só por erro pontual

**O que fiz:** além de MAE/RMSE/Spearman, uma tabela de decis com lift e participação no valor total.

**Por quê:** nenhum programa de retenção gasta no cliente médio. Gasta numa lista ordenada. Um modelo
pode ter MAE medíocre e ainda assim ordenar bem o suficiente para valer a pena — e o inverso também
acontece, que foi exatamente o que apareceu aqui.

**Detalhe de implementação:** ranqueio antes de binar. Se aplicasse `qcut` direto nas predições,
um modelo que prevê o mesmo valor para muitos clientes colapsaria decis inteiros. Tem teste pra isso.

---

## 5 perguntas prováveis, com resposta

### 1. "O LightGBM perdeu para um modelo de 1987. Como você explica isso?"

Não é que gradient boosting seja ruim, é que o problema não favorece ele.

São 3.478 clientes de treino, sete features e um alvo com cauda pesadíssima — o máximo de receita no
holdout é £168 mil contra mediana de £92. Com esse volume e essa distribuição, um modelo flexível
aprende a regularidade e erra feio na cauda.

O BG/NBD não aprende a relação, ele **assume** uma: existe um processo de compra e um processo de
abandono, ambos com heterogeneidade entre clientes. Quando essa estrutura bate com o processo gerador
real — e num varejo de recompra ela bate — a premissa funciona como regularização muito forte.

A falha do LightGBM é visível e específica: o decil 10 dele vale mais que os decis 5 a 9. Ele prevê
£24 médios para clientes que valem £414. Aprendeu que histórico magro significa gasto baixo, o que é
verdade na média e errado na cauda. O BG/NBD cai monotonicamente de 5,09x para 0,12x, sem inversão.

E a ressalva honesta: isso é específico desse tamanho de problema. Com 100x mais clientes eu esperaria
o LightGBM fechar e passar.

### 2. "Por que não deu features melhores ao LightGBM?"

Porque aí a comparação deixaria de ser entre os dois modelos e passaria a ser entre dois conjuntos de
informação. Os dois recebem a mesma base RFM.

Mas essa é exatamente a crítica certa ao experimento, e está no README como próximo passo: o dataset
tem país, categoria de produto e canal, e o BG/NBD **não tem como aceitar** nada disso. É a vantagem
estrutural do modelo supervisionado, e a comparação justa seguinte é dar essas features só pra ele e
ver se vira o jogo.

### 3. "O que é recency nesse modelo?"

É idade na última compra — dias entre a primeira e a última compra do cliente. **Não** é "dias desde a
última compra", que é o que a palavra significa numa segmentação RFM clássica.

Essa é a confusão mais comum com BG/NBD, e ela é silenciosa: se você inverter, o modelo passa a achar
que cliente recente é cliente velho, e a estimativa de quem está vivo inverte junto. Não dá erro, só
dá resultado errado. Tem teste travando a convenção.

### 4. "Por que MCMC e não MAP? MAP seria muito mais rápido."

Seria, e para um ponto de estimativa bastaria. Rodei MCMC porque a saída aqui direciona gasto em
**indivíduos**, e nesse caso eu quero o intervalo, não só a média.

A diferença prática: "esse cliente vale £800" e "esse cliente vale £800 com intervalo de £120 a
£3.000" levam a decisões diferentes sobre quanto investir em reter ele.

Dito isso, hoje o relatório só usa a média da posterior. Ter a distribuição e jogar fora a incerteza é
desperdício, e está no README como próximo passo: incorporar a probabilidade de o cliente já ter
abandonado na decisão de contato.

### 5. "Esse resultado se generaliza?"

Para negócios com esse formato de compra, sim. Para todo negócio, não — e eu tenho o contraexemplo no
próprio portfólio.

Essa base tem 67,3% de recompra e retenção que estabiliza em 15-20% ao mês. No `unit-economics-olist`,
que é marketplace brasileiro, **97% dos clientes compram uma única vez**. Lá, ajustar BG/NBD seria
desperdício: não há processo de recompra para modelar, e a conclusão correta de negócio é que o CAC
precisa se pagar no primeiro pedido.

Inclusive foi por isso que troquei o dataset: a especificação original sugeria Olist, e eu testei e vi
que com 3% de recompra o modelo não teria sinal. Escolher a base certa para a pergunta é parte do
trabalho.

---

## Números para ter na ponta da língua

| | |
| --- | --- |
| Clientes (após limpeza) | 5.878 |
| Clientes na calibração | 4.969 (3.478 treino / 1.491 teste) |
| Taxa de recompra | 67,3% |
| Ativos no holdout | 52,2% |
| Horizonte de previsão | 183 dias |
| MAE BG/NBD vs LightGBM | £484 vs £645 |
| Spearman BG/NBD vs LightGBM | 0,601 vs 0,485 |
| Lift do decil 1 | 5,09x vs 4,26x |
| Valor capturado no decil 1 | 51,2% vs 42,8% |
| Inflação se contasse invoice em vez de dia | 11,7% |

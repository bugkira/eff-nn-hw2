---
marp: true
size: 16:9
paginate: true
math: katex
style: |
  section {
    font-family: "IBM Plex Sans", "DejaVu Sans", "Noto Sans", sans-serif;
    color: #1a2332;
    background: #fafbfc;
    padding: 36px 48px !important;
    font-size: 21px;
    display: flex !important;
    flex-direction: column !important;
    justify-content: center !important;
    align-content: center !important;
    place-content: center !important;
    align-items: stretch !important;
    gap: 0.35em;
    box-sizing: border-box;
  }
  h1 {
    color: #0f2744;
    font-size: 28px;
    font-weight: 700;
    line-height: 1.2;
    margin: 0 0 0.12em 0;
    flex: 0 0 auto;
  }
  h2 {
    color: #1f4e79;
    font-size: 20px;
    margin: 0.15em 0 0.08em;
    font-weight: 700;
  }
  p { margin: 0.15em 0; }
  strong { color: #0f2744; }
  code, pre {
    font-family: "IBM Plex Mono", "DejaVu Sans Mono", monospace;
    font-size: 0.9em;
  }
  footer { color: #6b7280; font-size: 13px; }
  section.title {
    background: linear-gradient(160deg, #0f2744 0%, #1f4e79 55%, #2e75b6 100%);
    color: #f8fafc;
    padding: 48px 56px !important;
    justify-content: center !important;
    place-content: center !important;
    gap: 0.5em;
  }
  section.title h1 { color: #fff; font-size: 44px; margin: 0; line-height: 1.15; }
  section.title .lead {
    color: #fff;
    font-size: 24px;
    font-weight: 500;
    max-width: 94%;
    line-height: 1.35;
    margin: 0;
  }
  section.title .meta {
    color: #dbe7f5;
    font-size: 20px;
    margin: 0.35em 0 0;
    line-height: 1.5;
  }
  .muted { color: #6b7280; font-size: 16px; margin-top: 0.55em !important; }
  ul, ol {
    line-height: 1.35;
    margin: 0.1em 0;
    padding-left: 1.15em;
  }
  li { margin: 0.12em 0; }
  table {
    display: table !important;
    width: 100% !important;
    max-width: 100% !important;
    margin: 0.35em 0 !important;
    border-collapse: collapse !important;
    border-spacing: 0;
    table-layout: fixed !important;
    overflow: visible !important;
    font-size: 19px;
    flex: 0 0 auto;
    align-self: stretch;
  }
  th {
    background: #1f4e79;
    color: #fff;
    font-weight: 600;
    text-align: left;
  }
  td {
    background: #fff;
    border-bottom: 1px solid #d5dde8;
  }
  tr:nth-child(even) td { background: #f3f6fa; }
  td, th {
    padding: 0.45em 0.85em;
    vertical-align: middle;
    word-wrap: break-word;
    overflow-wrap: break-word;
  }
  td:first-child, th:first-child { text-align: left !important; }
  table.fullwidth {
    width: 100% !important;
    max-width: 100% !important;
    table-layout: fixed !important;
    margin: 0.3em 0 !important;
    align-self: stretch !important;
    font-size: 17px;
  }
  table.fullwidth td:first-child,
  table.fullwidth th:first-child {
    white-space: nowrap;
  }
  table.fullwidth code {
    font-size: 0.9em;
    padding: 0.02em 0.2em;
  }
  .cols {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45em 1.0em;
    width: 100%;
    flex: 0 0 auto;
    align-items: start;
  }
  .cols > div {
    background: transparent;
    border-radius: 0;
    padding: 0;
    font-size: 18px;
  }
  .cols table {
    font-size: 16px;
    margin: 0.25em 0 !important;
    max-width: 100% !important;
  }
  .cols p, .cols li { font-size: 17px; margin: 0.1em 0; }
  .cols h2 { font-size: 18px; margin: 0 0 0.2em; }
  .box {
    background: #eef3f8;
    border-left: 5px solid #1f4e79;
    padding: 0.6em 0.95em;
    margin: 0.2em 0;
    font-size: 21px;
    width: 100%;
    box-sizing: border-box;
    flex: 0 0 auto;
  }
  .katex-display { margin: 0.35em 0 !important; }
  p .katex { font-size: 1.06em; }
  .eqrow {
    display: grid;
    grid-template-columns: 1fr 2.4em;
    align-items: center;
    column-gap: 0.6em;
    width: 100%;
    margin: 0.25em 0;
    flex: 0 0 auto;
  }
  .eqrow .eqmath {
    min-width: 0;
    overflow-x: auto;
  }
  .eqrow .eqmath .katex-display {
    margin: 0.2em 0 !important;
  }
  .eqrow .eqno {
    font-size: 1.05em;
    color: #1a2332;
    text-align: right;
    line-height: 1.2;
    user-select: none;
  }
  table strong { color: #0f2744; font-weight: 700; }
  .takeaway {
    margin: 0.35em 0 0 0 !important;
    font-size: 20px;
    color: #0f2744;
    font-weight: 600;
  }
  .steps {
    margin: 0.25em 0;
    padding-left: 1.2em;
    font-size: 20px;
  }
  .steps li { margin: 0.2em 0; }
---

<!-- _class: title -->

# ParaRNN

<p class="lead">Параллельное обучение нелинейных RNN без линеаризации скрытого состояния</p>

<p class="meta">
<strong style="color:#fff">Danieli et al. · Apple · ICLR 2026</strong> · arXiv:2510.21450<br>
Efficient Neural Networks · Университет ИТМО · ДЗ-2
</p>

---

# Последовательная развёртка RNN

На этапе вывода классические LSTM и GRU привлекательны компактным скрытым состоянием $O(1)$, константным временем генерации токена и отсутствием KV-кэша. Основной барьер возникает при обучении: пошаговая зависимость (1) задаёт временную глубину $O(L)$. Вычисление $h_l$ строго требует результата $h_{l-1}$, что препятствует эффективному распараллеливанию на GPU.

<div class="eqrow">
<div class="eqmath">

$$
h_l = f(h_{l-1}, x_l),\qquad l = 1, \dots, L,\quad h_0 = 0
$$

</div>
<div class="eqno">(1)</div>
</div>

В отличие от подходов, переходящих к механизму внимания или упрощающих рекуррентность до линейной (как в SSM), ParaRNN сохраняет выразительную нелинейную функцию $f$. Вместо изменения архитектуры ячейки здесь распараллеливается само численное нахождение всей траектории $H$ во времени.

---

# Линейные SSM и associative scan

Чтобы преодолеть барьер $O(L)$ при обучении, современные State Space Models (SSM) упрощают переход между шагами до строго линейного:

<div class="eqrow">
<div class="eqmath">

$$
h_l = A_l h_{l-1} + B_l x_l
$$

</div>
<div class="eqno">(2)</div>
</div>

Линейные отображения ассоциативны относительно композиции. Это позволяет заменить последовательную развёртку алгоритмом parallel scan (prefix sum), сократив вычислительную глубину до логарифмической:

| | Последовательный проход | Parallel scan (SSM) |
|--|--|--|
| Вычислительная глубина | $O(L)$ | $O(\log L)$ |
| Суммарный объём работы | $O(L)$ | $O(L)$ |
| Требование к шагу перехода | Произвольный ($f$) | Строгая ассоциативность (линейность) |

## Цена линейности

Линейность по состоянию — вынужденная мера ради параллелизации на GPU. Платой за неё становится потеря выразительности: теоретически показано, что чисто линейные рекуррентности не способны моделировать сложные пошаговые зависимости и задачи с отслеживанием состояния (например, подсчёт чётности / *parity*).

<p class="takeaway">

Задача ParaRNN — вернуть скрытому состоянию нелинейность, сохранив параллельное обучение.

</p>

---

# Развёртка как $F(H)=0$

Вместо пошагового вычисления по времени вся траектория скрытых состояний $H=[h_1,\dots,h_L]^\top$ рассматривается как корень единой системы из $L$ нелинейных уравнений:

<div class="eqrow">
<div class="eqmath">

$$
F_l(H) = h_l - f(h_{l-1}, x_l) = 0,\qquad l = 1,\dots,L
$$

</div>
<div class="eqno">(3)</div>
</div>

где граничное условие $h_0=0$ делает систему замкнутой и однозначно разрешимой ($L$ уравнений для $L$ неизвестных).

Суть подхода авторов:

- **Итерации метода Ньютона.** Вместо последовательного цикла по времени система решается численно — через последовательную линеаризацию невязок $F(H)$.
- **Марковская структура.** Поскольку шаг $l$ зависит только от текущего $h_l$ и предыдущего $h_{l-1}$, глобальный якобиан системы получается блочно-бидиагональным. Это позволяет не хранить плотную матрицу размера $(L\cdot d_h)^2$.
- **Параллельная инициализация.** Стартовая точка $h_l^{(0)}=f(0,x_l)$ вычисляется независимо для всех токенов за один параллельный проход по последовательности.

---

# Ньютон и parallel scan

На каждой итерации $k$ система невязок линеаризуется вокруг текущей траектории $H^{(k)}$:

<div class="eqrow">
<div class="eqmath">

$$
J_F(H^{(k)})\,\Delta H = -F(H^{(k)}),\qquad H^{(k+1)} = H^{(k)} + \Delta H
$$

</div>
<div class="eqno">(4)</div>
</div>

Поскольку глобальный якобиан $J_F$ блочно-бидиагонален, строчка $l$ матричной системы раскрывается в локальное уравнение для поправок:

<div class="eqrow">
<div class="eqmath">

$$
\Delta h_l = J_f\big|_{h_{l-1}}\,\Delta h_{l-1} + r_l,
\qquad r_l = f(h_{l-1}, x_l) - h_l
$$

</div>
<div class="eqno">(5)</div>
</div>

**Главный инсайт авторов.** Уравнение поправок (5) — линейная рекуррентность:

- локальный якобиан $J_f$ играет роль матрицы перехода ($A_l$);
- невязка $r_l$ играет роль входного вектора ($B_l x_l$).

Линейная зависимость ассоциативна, поэтому вектор поправок $\Delta H$ находится не последовательным проходом, а алгоритмом parallel scan за $O(\log L)$.

<p class="takeaway">

Итоговая схема: внешний решатель (метод Ньютона) выполняет несколько итераций уточнения, а внутренний решатель на каждом шаге распараллеливается по всей длине последовательности на GPU.

</p>

---

# Сходимость метода Ньютона

## Условие выигрыша по времени

Параллельный солвер быстрее последовательного прохода $O(L)$, пока выполняется условие:

$$
N_{\mathrm{its}} \ll \frac{L}{\log_2 L}
$$

Это даёт большой запас: даже при сублинейном числе итераций (например, $N_{\mathrm{its}}\sim O(\log L)$) глубина $\mathcal{O}(\log^2 L)$ асимптотически обгоняет последовательную развёртку.

## Эмпирическая сходимость в статье (рис. 4)

На практике этот запас почти не расходуется — невязка затухает практически мгновенно:

- максимальная невязка $\|F\|_\infty$ падает до машинной точности ($10^{-5}$–$10^{-6}$) за 3 шага;
- скорость сходимости инвариантна к длине: кривые для $L=2^{8}$ и $L=2^{11}$ накладываются друг на друга;
- поведение одинаково устойчиво на случайных весах и на обученных моделях до 7B.

Для обучения фиксируется константный бюджет $N_{\mathrm{its}}=3$.

## Оговорка

Сходимость за 3 шага обеспечена диагональной структурой ячеек и сжатием весов ($|a_*|\le 0{,}5$). Для произвольных новых RNN поведение солвера потребует отдельной валидации.

---

# Диагонализация матриц перехода: путь к $\mathcal{O}(d_h)$

## В чём проблема исходных ячеек

В стандартных LSTM и GRU скрытые состояния смешиваются плотными матрицами. В параллельном скане это потребовало бы перемножать блоки $d_h\times d_h$ вдоль всей цепочки — схема становится непрактичной: $\mathcal{O}(L d_h^2)$ памяти и $\mathcal{O}(d_h^3)$ операций на каждый шаг редукции.

## Решение: диагональные веса ячейки

Авторы делают диагональными сами параметры перехода и peephole-связи:

<div class="eqrow">
<div class="eqmath">

$$
A_*=\operatorname{diag}(a_*),\qquad C_*=\operatorname{diag}(c_*),\qquad a_*,c_*\in\mathbb{R}^{d_h}
$$

</div>
<div class="eqno">(6)</div>
</div>

Это даёт строго диагональный якобиан у ParaGRU и блочно-диагональный $2\times 2$ у ParaLSTM (из-за пары состояний $c_l$ и $h_l$), уменьшая расходы памяти с $\mathcal{O}(L d_h^2)$ до $\mathcal{O}(L d_h)$ и критически снижая сложность вычислений с $\mathcal{O}(d_h^3)$ до $\mathcal{O}(d_h)$.

## Уязвимость в аргументации авторов

- Диагонализация превращает $d_h$-мерную ячейку в $d_h$ изолированных одномерных рекуррентностей — каналы во времени больше не взаимодействуют напрямую.
- Авторы утверждают, что смешивание каналов компенсируют внешние слои MLP. Но выразительность они исследовали ограниченно: на синтетических задачах проверена лишь простая $\mathbb{Z}_2$-Parity, для которой межканальное взаимодействие не нужно (достаточно одного скалярного регистра).
- Вопрос о том, способна ли связка «1D-нелинейность + послойный MLP» заменить полноценную многомерную нелинейную динамику, в статье остаётся открытым.

---

# Обратный проход: один скан вместо метода Ньютона

Дифференцирование рекуррентности (BPTT) даёт строго линейную цепочку по градиентам скрытого состояния:

<div class="eqrow">
<div class="eqmath">

$$
\nabla_{h_{l-1}}\mathcal{L}
=
\big(J_f|_{h_{l-1}}\big)^\top
\nabla_{h_l}\mathcal{L}
+
\partial_{h_{l-1}}\mathcal{L},
\qquad l = L,\dots,1
$$

</div>
<div class="eqno">(7)</div>
</div>

с граничным условием $\nabla_{h_L}\mathcal{L}=\partial_{h_L}\mathcal{L}$.

Поскольку система линейна, итерации метода Ньютона не требуются, и все градиенты $\nabla_h\mathcal{L}$ находятся за один обратный parallel scan.

Бюджет вычислений на полный шаг обучения (Forward + Backward):

- Mamba: $1$ (fwd) + $1$ (bwd) = **$2$** скана
- ParaRNN: $3$ (fwd, метод Ньютона) + $1$ (bwd) = **$4$** скана

<p class="takeaway">

Суммарные накладные расходы нелинейности на шаге обучения составляют всего $\sim 2\times$ относительно чисто линейных SSM.

</p>

<!-- notes
Сравнение с Mamba в статье — для упрощённого модуля (без полной обвязки).
-->

---

# Реализация на GPU: аппаратный parallel scan

Чтобы обеспечить вычислительную эффективность, авторы разработали собственное CUDA-ядро, раскладывающее параллельную редукцию по всей аппаратной иерархии GPU:

<table class="fullwidth">
<thead>
<tr><th style="width:24%">Уровень</th><th>Механика</th></tr>
</thead>
<tbody>
<tr><td>Thread (registers)</td><td>Локальный chunk 2–4 уравнений (<code>chunk_size</code>), прямое исключение</td></tr>
<tr><td>Warp (registers)</td><td>Обмен через <code>__shfl_down_sync</code>, без обращения к памяти</td></tr>
<tr><td>Block (Shared memory)</td><td>Сведение результатов варпов с обратной подстановкой</td></tr>
<tr><td>Grid (Global memory)</td><td>При $L \ge 2^{16}$ — сетка блоков с синхронизацией через VRAM</td></tr>
</tbody>
</table>

## Уровни интеграции в ParaRNN

- **Modular (PyTorch + CUDA):** сборка якобиана выполняется в PyTorch (через autograd или аналитически), сам скан — в CUDA. Это сохраняет гибкость при создании новых типов ячеек.
- **Fully-fused CUDA:** весь цикл метода Ньютона (невязки, сборка якобиана и скан) объединён в одно ядро. Это устраняет накладные расходы на промежуточный обмен данными через VRAM.

## Производительность (A100, $L=2^{9}$)

Ускорение до $\sim 665\times$ относительно последовательной развёртки ячейки. В режиме Fully-fused время исполнения ячейки в $1{,}5\times$ (ParaLSTM) – $2{,}6\times$ (ParaGRU) меньше, чем у Mamba.

---

# Экспериментальные результаты

Авторы валидируют ParaRNN по двум направлениям: выразительность рекуррентной ячейки на синтетических тестах и качество при масштабировании языковой модели до 7B параметров.

| Модель | MQAR | $k$-hop | Parity | PPL↓ (7B) |
|--|--:|--:|--:|--:|
| Transformer | 100% | 78% | ~53% | 9,55 |
| Mamba2 | 100% | ~98% | ~51% | **8,62** |
| **ParaLSTM** | **100%** | **100%** | **100%** | 9,16 |
| **ParaGRU** | **100%** | **100%** | **100%** | 9,19 |

<p class="caption" style="text-align:center;font-size:15px;color:#4b5563;margin:0.1em 0 0.35em;">
Синтетические тесты (1 слой) · SlimPajama 7B
</p>

## Ключевые выводы

- **Выразительность нелинейности.** На задачах отслеживания состояния ($k$-hop) и периодической логики (Parity) чисто линейная Mamba2 падает до уровня случайного угадывания (~51%), тогда как ParaGRU и ParaLSTM решают их на 100%.
- **Масштабирование до 7B.** При обучении по законам масштабирования Chinchilla нелинейные RNN обходят baseline Transformer по перплексии (9,16–9,19 vs 9,55) и показывают устойчивую сходимость градиентов без деградации качества.

---

# Архитектура и код

Для тестов авторы адаптировали трансформерный стек DCLM: механизм внимания заменили на блок ParaRNN (ParaGRU/ParaLSTM) с каузальной свёрткой и гейтированием в стиле Mamba, оставив перемешивание каналов внешним слоям MLP. Модели от 400M до 7B параметров обучались на SlimPajama (без Books3) по законам масштабирования Chinchilla.

Фреймворк `apple/ml-pararnn` автоматизирует параллелизацию: достаточно описать шаг $f$ и аналитический якобиан $J_f$, а сборку уравнений и параллельный скан библиотека выполняет сама. На этапе вывода модель переходит в стандартный рекуррентный режим без итераций метода Ньютона, сохраняя преимущество классических RNN — работу с памятью $O(1)$ без KV-кэша.

---

# Воспроизведение

Для воспроизведения статьи разработана библиотека `pararnn-torch` (`github.com/bugkira/pararnn-torch`). На обучении она считает всю траекторию скрытых состояний методом Ньютона со сканом; в режиме вывода остаётся тот же переход $f$, но уже по одному шагу за раз. Согласие параллельного и последовательного проходов проверяется штатным вызовом API этой же библиотеки.

В численных экспериментах статьи заданы такие параметры:

| Параметр | В статье |
|--|--|
| Init | $h_l^{(0)}=f(0,x_l)$ |
| Ньютон | $N_{\mathrm{its}}=3$ |
| $\|a_*\|$ | ≤ 0,5 на текстовых данных, ≤ 0,9 на синтетических задачах |

---

# Выводы

Нелинейную развёртку RNN можно эффективно распараллелить, если искать всю траекторию состояний сразу как корень системы $F(H)=0$ методом Ньютона. Ключевое инженерное условие такой схемы — диагонализация матриц перехода $A_*$: она меняет асимптотику сложности одного шага скана с кубической $\mathcal{O}(d_h^3)$ на линейную $\mathcal{O}(d_h)$ и делает вычисления на GPU сопоставимыми по стоимости с линейными SSM.

На практике для ParaGRU и ParaLSTM уже трёх шагов метода Ньютона достаточно, чтобы параллельное приближение совпало с точной пошаговой развёрткой. В результате классические рекуррентные ячейки масштабируются до 7B без потери качества на языковом моделировании, но сохраняют нелинейную динамику там, где линейные модели систематически не справляются — например, на задаче Parity.

<p class="muted">
`pararnn-torch` · `apple/ml-pararnn` · arXiv:2510.21450<br>
Danieli et al., ICLR 2026 · ИТМО · Efficient NN · ДЗ-2
</p>

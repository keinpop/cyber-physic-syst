from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import requests

# Конфигурация

OLLAMA_BASE_URL: str = "http://localhost:11434"
MODEL_NAME: str = "qwen2.5:0.5b"

QUERIES: list[str] = [
    "Что такое машинное обучение? Объясни кратко.",
    "Напиши функцию на Python для вычисления факториала числа.",
    "Какова столица Франции? Ответь одним предложением.",
    "Объясни рекурсию простыми словами, как будто мне 10 лет.",
    "Назови три преимущества регулярных физических упражнений.",
    "Напиши сказку о весне в 5 предложений.",
    "Что такое протокол HTTP? Кратко поясни принцип работы.",
    "В чём разница между списком (list) и кортежем (tuple) в Python?",
    "Какова временная сложность алгоритма бинарного поиска?",
    "Переведи на английский: «Привет, как у тебя дела?»",
]

"""Проверяет доступность Ollama-сервера.

Отправляет GET-запрос на корневой эндпоинт и возвращает True,
если сервер отвечает HTTP 200.

Args:
    base_url: Базовый URL Ollama-сервера.

Returns:
    True — сервер доступен, False — недоступен.
"""
def check_server(base_url: str = OLLAMA_BASE_URL) -> bool:
    try:
        r = requests.get(base_url, timeout=5)
        return r.status_code == 200
    except requests.ConnectionError:
        return False



"""Отправляет один запрос к Ollama /api/generate и возвращает ответ модели.

  Использует stream=False, чтобы получить полный ответ одним JSON-объектом
  (без серверного стриминга по токенам).

  Args:
      prompt:   Текст запроса для LLM.
      model:    Название модели в Ollama (например, "qwen2.5:0.5b").
      base_url: Базовый URL Ollama-сервера.
      timeout:  Таймаут HTTP-запроса в секундах.

  Returns:
      Строка с ответом модели (strip применён).

  Raises:
      requests.HTTPError:       Сервер вернул код 4xx/5xx.
      requests.ConnectionError: Сервер недоступен.
      requests.Timeout:         Сервер не ответил за timeout секунд.
      KeyError:                 JSON-ответ не содержит поля "response".
"""
def generate(
    prompt: str,
    model: str = MODEL_NAME,
    base_url: str = OLLAMA_BASE_URL,
    timeout: int = 120,
) -> str:
    payload = {"model": model, "prompt": prompt, "stream": False}
    resp = requests.post(f"{base_url}/api/generate", json=payload, timeout=timeout)
    resp.raise_for_status()
    return resp.json()["response"].strip()


"""Прогоняет список запросов через LLM и собирает результаты.

Каждый запрос обрабатывается последовательно. В консоль выводится
прогресс и время выполнения каждого запроса.

Args:
    queries: Список текстовых запросов.
    model:   Название модели Ollama.

Returns:
    Список словарей с ключами:
      • "n"         (int)   — порядковый номер, начиная с 1
      • "query"     (str)   — исходный запрос
      • "response"  (str)   — ответ модели
      • "elapsed_s" (float) — время генерации в секундах
"""
def run_inference(queries: list[str], model: str = MODEL_NAME) -> list[dict]:
    results: list[dict] = []
    total = len(queries)

    for n, query in enumerate(queries, start=1):
        print(f"[{n:2d}/{total}] {query[:70]}...")
        t0 = time.perf_counter()
        response = generate(query, model=model)
        elapsed = round(time.perf_counter() - t0, 2)
        print(f"       ↳ {elapsed}s | {response[:90].replace(chr(10), ' ')}\n")
        results.append({"n": n, "query": query, "response": response, "elapsed_s": elapsed})

    return results

def _html_escape(text: str) -> str:
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
    )


def save_html(
    results: list[dict],
    model: str = MODEL_NAME,
    path: str | Path = "inference_report.html",
) -> None:
    ts = datetime.now().strftime("%d.%m.%Y %H:%M")
    avg_t = round(sum(r["elapsed_s"] for r in results) / len(results), 2)

    rows_html = ""
    for r in results:
        q = _html_escape(r["query"])
        a = _html_escape(r["response"]).replace("\n", "<br>")
        rows_html += (
            f'<tr>'
            f'<td class="num">{r["n"]}</td>'
            f'<td class="query">{q}</td>'
            f'<td class="answer">{a}</td>'
            f'<td class="time">{r["elapsed_s"]}s</td>'
            f'</tr>\n'
        )

    html = f"""<!DOCTYPE html>
<html lang="ru" data-theme="light">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Inference Report — {model}</title>
<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600&family=Inter:wght@300..700&display=swap" rel="stylesheet">
<style>
:root,[data-theme=light]{{
  --bg:#f7f6f2;--surface:#ffffff;--surface-2:#f3f0ec;
  --border:rgba(40,37,29,.10);--text:#28251d;--muted:#7a7974;--faint:#bab9b4;
  --accent:#01696f;--accent-hi:#cedcd8;--accent-text:#fff;
  --code-bg:#f0efeb;--shadow:0 2px 8px rgba(40,37,29,.08);
  --radius:0.5rem;
}}
[data-theme=dark]{{
  --bg:#171614;--surface:#1c1b19;--surface-2:#22211f;
  --border:rgba(205,204,202,.10);--text:#cdccca;--muted:#797876;--faint:#5a5957;
  --accent:#4f98a3;--accent-hi:#313b3b;--accent-text:#171614;
  --code-bg:#201f1d;--shadow:0 2px 8px rgba(0,0,0,.30);
}}
*,*::before,*::after{{box-sizing:border-box;margin:0;padding:0}}
html{{scroll-behavior:smooth;-webkit-font-smoothing:antialiased}}
body{{background:var(--bg);color:var(--text);font-family:'Inter',sans-serif;
  font-size:clamp(0.9rem,.85rem + .25vw,1rem);line-height:1.6;min-height:100dvh}}
/* ── header ── */
header{{
  background:var(--surface);border-bottom:1px solid var(--border);
  padding:1.25rem 2rem;display:flex;align-items:center;
  justify-content:space-between;gap:1rem;position:sticky;top:0;z-index:10;
  box-shadow:var(--shadow);
}}
.logo{{display:flex;align-items:center;gap:.6rem;font-weight:700;font-size:1rem}}
.logo svg{{flex-shrink:0}}
.theme-btn{{background:none;border:1px solid var(--border);border-radius:var(--radius);
  padding:.35rem .6rem;cursor:pointer;color:var(--muted);font-size:.8rem;
  transition:color .18s,border-color .18s}}
.theme-btn:hover{{color:var(--text);border-color:var(--accent)}}
/* ── hero ── */
.hero{{padding:2.5rem 2rem 2rem;max-width:1100px;margin-inline:auto}}
.hero h1{{font-size:clamp(1.4rem,1.2rem + 1vw,2rem);font-weight:700;
  letter-spacing:-.02em;margin-bottom:.5rem}}
.hero p{{color:var(--muted);max-width:60ch}}
/* ── meta chips ── */
.meta{{display:flex;flex-wrap:wrap;gap:.5rem;margin-top:1.25rem}}
.chip{{background:var(--surface-2);border:1px solid var(--border);border-radius:9999px;
  padding:.25rem .75rem;font-size:.78rem;color:var(--muted);
  font-family:'JetBrains Mono',monospace}}
.chip b{{color:var(--text)}}
.chip.accent{{background:var(--accent-hi);border-color:transparent;color:var(--accent)}}
/* ── table ── */
.table-wrap{{max-width:1100px;margin:0 auto 3rem;padding:0 2rem;overflow-x:auto}}
table{{width:100%;border-collapse:collapse;background:var(--surface);
  border-radius:var(--radius);overflow:hidden;box-shadow:var(--shadow)}}
thead th{{background:var(--surface-2);font-size:.78rem;font-weight:600;
  text-transform:uppercase;letter-spacing:.06em;color:var(--muted);
  padding:.75rem 1rem;border-bottom:1px solid var(--border);text-align:left}}
tbody tr{{border-bottom:1px solid var(--border);transition:background .15s}}
tbody tr:last-child{{border-bottom:none}}
tbody tr:hover{{background:var(--surface-2)}}
td{{padding:.85rem 1rem;vertical-align:top;font-size:.9rem}}
.num{{width:2.5rem;font-variant-numeric:tabular-nums;color:var(--faint);
  font-family:'JetBrains Mono',monospace;font-size:.8rem}}
.query{{width:32%;font-weight:500;color:var(--text)}}
.answer{{color:var(--muted);line-height:1.65}}
.answer code,.answer pre{{background:var(--code-bg);padding:.15em .35em;
  border-radius:.25rem;font-family:'JetBrains Mono',monospace;font-size:.82em}}
.time{{width:4.5rem;font-family:'JetBrains Mono',monospace;font-size:.78rem;
  color:var(--faint);white-space:nowrap;text-align:right}}
/* ── footer ── */
footer{{border-top:1px solid var(--border);text-align:center;
  padding:1.5rem;color:var(--faint);font-size:.78rem}}
/* ── mobile ── */
@media(max-width:640px){{
  header{{padding:1rem}}
  .hero,.table-wrap{{padding-inline:1rem}}
  .query{{width:40%}}
  .time{{display:none}}
}}
</style>
</head>
<body>
<header>
  <div class="logo">
    <svg width="28" height="28" viewBox="0 0 28 28" fill="none" aria-label="Lab logo">
      <rect width="28" height="28" rx="6" fill="var(--accent)" opacity=".15"/>
      <path d="M7 21L11 7h2l2 9 2-5h3" stroke="var(--accent)" stroke-width="1.8"
            stroke-linecap="round" stroke-linejoin="round"/>
      <circle cx="21" cy="19" r="2" fill="var(--accent)"/>
    </svg>
    <span>Inference Report</span>
  </div>
  <button class="theme-btn" data-theme-toggle aria-label="Switch theme">☀ / ☾</button>
</header>

<main>
  <section class="hero">
    <h1>Отчёт инференса LLM</h1>
    <p>Результаты 10 запросов к модели <strong>{model}</strong> через Ollama HTTP API.</p>
    <div class="meta">
      <span class="chip accent"><b>Модель:</b> {model}</span>
      <span class="chip"><b>Дата:</b> {ts}</span>
      <span class="chip"><b>Запросов:</b> {len(results)}</span>
      <span class="chip"><b>Среднее время:</b> {avg_t}s</span>
      <span class="chip"><b>Эндпоинт:</b> {OLLAMA_BASE_URL}/api/generate</span>
    </div>
  </section>

  <div class="table-wrap">
    <table>
      <thead>
        <tr>
          <th>#</th>
          <th>Запрос к LLM</th>
          <th>Вывод LLM</th>
          <th>Время</th>
        </tr>
      </thead>
      <tbody>
        {rows_html}
      </tbody>
    </table>
  </div>
</main>

<footer>Лабораторная работа 2: NLP · Кибер-физические системы · 2026</footer>

<script>
(function(){{
  const btn=document.querySelector('[data-theme-toggle]'),h=document.documentElement;
  let d=matchMedia('(prefers-color-scheme:dark)').matches?'dark':'light';
  h.setAttribute('data-theme',d);
  btn&&btn.addEventListener('click',()=>{{
    d=d==='dark'?'light':'dark';
    h.setAttribute('data-theme',d);
  }});
}})();
</script>
</body>
</html>"""

    Path(path).write_text(html, encoding="utf-8")
    print(f"✓ HTML → {path}")


def main() -> None:
    print(f"Модель  : {MODEL_NAME}")
    print(f"Сервер  : {OLLAMA_BASE_URL}\n")

    if not check_server():
        print("ОШИБКА: Ollama-сервер недоступен.")
        print("→ Запустите:  ollama serve")
        raise SystemExit(1)

    print("Сервер  : OK\n" + "─" * 60)
    results = run_inference(QUERIES)
    print("─" * 60)

    save_html(results)

    total = round(sum(r["elapsed_s"] for r in results), 1)
    print(f"\nГотово. Запросов: {len(results)} | Суммарно: {total}s")


if __name__ == "__main__":
    main()
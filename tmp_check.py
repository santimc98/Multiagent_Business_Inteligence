from src.graph.graph import dialect_guard_violations
code = "import pandas as pd\npd.read_csv('data/raw.csv', sep=',', decimal='.', encoding='utf-8')\n"
print(dialect_guard_violations(code, ';', ',', 'utf-8'))

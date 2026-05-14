with open('test_calc_winrate.py', 'r') as f:
    lines = f.readlines()

new_lines = []
for line in lines:
    if "from calc_winrate import get_score_at_75" in line:
        new_lines.append(line.rstrip() + "  # noqa: E402\n")
    elif "import pytest" in line:
        new_lines.append(line.rstrip() + "  # noqa: E402\n")
    else:
        new_lines.append(line)

with open('test_calc_winrate.py', 'w') as f:
    f.writelines(new_lines)

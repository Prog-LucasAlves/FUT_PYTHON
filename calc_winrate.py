import pandas as pd


def get_bin(val, bins, labels):
    if pd.isna(val):
        return None
    for i in range(len(bins) - 1):
        if bins[i] < val <= bins[i + 1]:
            return labels[i]
    return labels[-1]


def get_team_averages(df):
    home_goals = df.groupby("Home")["Goals_H_FT"].sum()
    home_games = df.groupby("Home")["Home"].count()
    away_goals = df.groupby("Away")["Goals_A_FT"].sum()
    away_games = df.groupby("Away")["Away"].count()

    total_goals = home_goals.add(away_goals, fill_value=0)
    total_games = home_games.add(away_games, fill_value=0)

    avg_goals = (total_goals / total_games).to_dict()
    return avg_goals


def get_score_at_75(min_str):
    if pd.isna(min_str) or min_str == "" or min_str == "[]":
        return 0
    if isinstance(min_str, str):
        min_str = min_str.replace("'", "").replace("[", "").replace("]", "").strip()
        if not min_str:
            return 0
        minutes = min_str.split(",")
    elif isinstance(min_str, list):
        minutes = min_str
    else:
        return 0

    count = 0
    for m in minutes:
        m = str(m).strip()
        if "+" in m:
            m = m.split("+")[0]
        try:
            m_val = int(m)
            if m_val <= 75:
                count += 1
        except ValueError:
            pass
    return count


if __name__ == "__main__":
    df = pd.read_csv("data_total/dados_historico.csv", sep=";", low_memory=False)

    team_avg_goals = get_team_averages(df)

    bins_h = [1.0, 1.3, 1.5, 1.7, 2.0, 2.5, 3.0, 100]
    labels_h = ["<1.3", "1.3-1.5", "1.5-1.7", "1.7-2.0", "2.1-2.5", "2.6-3.0", "3.0+"]

    bins_over = [0, 1.6, 1.8, 2.0, 100]
    labels_over = ["<1.6", "1.6-1.8", "1.8-2.0", "2.0+"]

    bins_btts = [0, 1.6, 1.8, 2.0, 100]
    labels_btts = ["<1.6", "1.6-1.8", "1.8-2.0", "2.0+"]

    bins_lay = [0, 10, 15, 20, 30, 100]
    labels_lay = ["<10", "10-15", "15-20", "20-30", "30+"]

    bins_avg_h = [0, 1.2, 1.5, 1.8, 5.0]
    labels_avg_h = ["<1.2", "1.2-1.5", "1.5-1.8", "1.8+"]

    winning_brackets = [
        ("2.1-2.5", "2.0+", "1.8-2.0", "10-15", "1.2-1.5"),
        ("1.7-2.0", "2.0+", "2.0+", "15-20", "1.2-1.5"),
        ("2.1-2.5", "1.6-1.8", "1.6-1.8", "15-20", "1.8+"),
        ("2.1-2.5", "1.8-2.0", "1.6-1.8", "10-15", "1.2-1.5"),
        ("<1.3", "<1.6", "1.8-2.0", "30+", "1.8+"),
        ("1.5-1.7", "2.0+", "2.0+", "15-20", "1.5-1.8"),
        ("1.3-1.5", "1.8-2.0", "2.0+", "20-30", "1.5-1.8"),
        ("3.0+", "1.8-2.0", "1.6-1.8", "10-15", "1.8+"),
        ("1.3-1.5", "<1.6", "1.8-2.0", "30+", "1.8+"),
        ("3.0+", "2.0+", "2.0+", "<10", "1.5-1.8"),
    ]

    total_selected = 0
    greens = 0
    reds_10 = 0
    reds_50 = 0

    total_profit = 0.0
    total_staked = 0.0

    for idx, row in df.iterrows():
        bin_h = get_bin(row.get("Odd_H_Back"), bins_h, labels_h)
        bin_over = get_bin(row.get("Odd_Over25_FT_Back"), bins_over, labels_over)
        bin_btts = get_bin(row.get("Odd_BTTS_Yes_Back"), bins_btts, labels_btts)
        bin_lay = get_bin(row.get("Odd_CS_0x1_Lay"), bins_lay, labels_lay)

        home_avg = team_avg_goals.get(row["Home"], 0)
        bin_avg_h = get_bin(home_avg, bins_avg_h, labels_avg_h)

        if (bin_h, bin_over, bin_btts, bin_lay, bin_avg_h) in winning_brackets:
            total_selected += 1

            odd_lay_val = row.get("Odd_CS_0x1_Lay", 0)
            if pd.isna(odd_lay_val) or odd_lay_val == 0:
                odd_lay_val = 15.0  # fallback average just in case

            liability = odd_lay_val - 1
            stake = 1.0
            total_staked += stake

            goals_h_75 = get_score_at_75(row.get("Goals_Min_H"))
            goals_a_75 = get_score_at_75(row.get("Goals_Min_A"))

            if goals_h_75 == 0 and goals_a_75 == 0:
                reds_10 += 1
                # Loss is 10% of liability
                total_profit -= 0.1 * liability
            elif goals_h_75 == 0 and goals_a_75 == 1:
                reds_50 += 1
                # Loss is 50% of liability
                total_profit -= 0.5 * liability
            else:
                greens += 1
                # Win is the stake
                total_profit += stake

    print(f"Total Selected Games: {total_selected}")
    print(f"Total Profit (Units): {total_profit:.2f}")
    print(f"ROI on Stake: {(total_profit / total_staked) * 100:.2f}%")

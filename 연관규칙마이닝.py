import csv
from itertools import combinations

def apriori(file_path, min_support=0.02, min_confidence=0.3):

    total_transactions = 0
    item_counts = {}
    
    # 첫 번째 패스: 1-항목 집합
    with open(file_path, 'r') as file:
        reader = csv.reader(file)
        for transaction in reader:
            total_transactions += 1
            items = frozenset(transaction)
            for item in items:
                if frozenset([item]) in item_counts:
                    item_counts[frozenset([item])] += 1
                else:
                    item_counts[frozenset([item])] = 1

    # min_support_count
    min_support_count = min_support * total_transactions

    # 1-빈번 항목 집합 필터링
    frequent_itemsets = {item: count for item, count in item_counts.items() if count >= min_support_count}

    rules = []

    k = 2
    while frequent_itemsets:
        candidate_itemsets = {}
        
        # 두 번째 패스부터: 트랜잭션 다시 읽어 k-항목 집합 생성
        with open(file_path, 'r') as file:
            reader = csv.reader(file)
            for transaction in reader:
                items = frozenset(transaction)
                if len(items) >= k:
                    for itemset in combinations(items, k):
                        itemset = frozenset(itemset)
                        if all(frozenset(subset) in frequent_itemsets for subset in combinations(itemset, k - 1)):
                            if itemset in candidate_itemsets:
                                candidate_itemsets[itemset] += 1
                            else:
                                candidate_itemsets[itemset] = 1

        # k-빈번 항목 집합 필터링
        frequent_itemsets = {item: count for item, count in candidate_itemsets.items() if count >= min_support_count}

        # 연관 규칙
        for itemset, count in frequent_itemsets.items():
            for consequent in itemset:
                antecedent = itemset - frozenset([consequent])
                if len(antecedent) > 0:
                    antecedent_count = item_counts.get(frozenset(antecedent), 0)
                    consequent_count = item_counts.get(frozenset([consequent]), 0)
                    if antecedent_count > 0:
                        conf = count / antecedent_count
                        lift = (count / total_transactions) / ((antecedent_count / total_transactions) * (consequent_count / total_transactions))
                        if conf >= min_confidence:
                            rules.append((antecedent, frozenset([consequent]), conf, lift))

        # 패스 증가
        k += 1

    print(f"연관 규칙:")
    for rule in rules:
        antecedent, consequent, conf, lift = rule
        print(f"{set(antecedent)} => {set(consequent)} (conf: {conf:.2f}, lift: {lift:.2f})")


file_path = 'market.csv'
min_support = 0.02
min_confidence = 0.3

apriori(file_path, min_support, min_confidence)

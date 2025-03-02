import pandas as pd
import numpy as np
import time
import itertools
import os
from mlxtend.frequent_patterns import apriori
from mlxtend.frequent_patterns import association_rules


# Data Processing Functions
# -------------------------
def read_data(items_file, transactions_file):

    items_df = pd.read_csv(items_file)
    items_dict = dict(zip(items_df['Item #'], items_df['Item Name']))
    
    transactions_df = pd.read_csv(transactions_file)
    
    transactions = []
    for _, row in transactions_df.iterrows():
        items = [item.strip() for item in row['Transaction'].split(',')]
        transactions.append(items)
    
    return items_dict, transactions


def create_one_hot_encoding(transactions, items_dict):
   
    transaction_data = []
    item_names = list(items_dict.values())
    
    for transaction in transactions:
        row = [1 if item in transaction else 0 for item in item_names]
        transaction_data.append(row)
    
    # Create DataFrame with item names as columns
    one_hot_df = pd.DataFrame(transaction_data, columns=item_names)
    
    return one_hot_df


# Brute Force Algorithm Implementation
# -----------------------------------
def get_support(itemset, transactions):

    count = 0
    for transaction in transactions:
        if all(item in transaction for item in itemset):
            count += 1
    return count / len(transactions)


def brute_force_frequent_itemsets(transactions, min_support=0.3):
 
    start_time = time.time()
    
    # unique items ( list --> set --> list)
    unique_items = set()
    for transaction in transactions:
        for item in transaction:
            unique_items.add(item)
    unique_items = list(unique_items)
    
    frequent_itemsets = {}
    
    # all possible k-itemsets
    k = 1
    while True:
        candidates = list(itertools.combinations(unique_items, k))
        
        # Checking frequent itemset for each candidate
        found_frequent = False
        for candidate in candidates:
            support = get_support(candidate, transactions)
            if support >= min_support:
                frequent_itemsets[candidate] = support
                found_frequent = True
        
        if not found_frequent:
            break
        
        k += 1
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return frequent_itemsets, execution_time


def generate_association_rules_brute_force(frequent_itemsets, min_confidence):

    start_time = time.time()
    
    rules = []
    
    # Generate rules from each frequent itemset
    for itemset, support in frequent_itemsets.items():
        if len(itemset) < 2:
            continue
        
        # Generating all possible antecedent/consequent splits
        for i in range(1, len(itemset)):
            for antecedent in itertools.combinations(itemset, i):
                consequent = tuple(item for item in itemset if item not in antecedent)
                
                # Skiping if empty
                if not antecedent or not consequent:
                    continue
                
                antecedent_support = frequent_itemsets.get(antecedent, 0)
                if antecedent_support == 0:
                    continue
                
                confidence = support / antecedent_support
                
                if confidence >= min_confidence:
                  
                    consequent_support = frequent_itemsets.get(consequent, 0)
                    if consequent_support == 0:
                        continue
                                                           
                    rules.append({
                        'antecedent': antecedent,
                        'consequent': consequent,
                        'support': support,
                        'confidence': confidence,
                        
                    })
    
    end_time = time.time()
    execution_time = end_time - start_time
    
    return rules, execution_time


# Apriori Implementation (using mlxtend)
# -------------------------------------
def run_apriori_mlxtend(one_hot_df, min_support=0.3, min_confidence=0.5):

    start_time = time.time()
    
    frequent_itemsets = apriori(one_hot_df, min_support=min_support, use_colnames=True)
    
    frequent_itemsets_time = time.time() - start_time
    
    rules_start_time = time.time()
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=min_confidence)
    
    rules_time = time.time() - rules_start_time
    
    total_time = time.time() - start_time
    
    return frequent_itemsets, rules, total_time,frequent_itemsets_time , rules_time


# Output Formatting Functions
# --------------------------
def format_itemset(itemset):

    if isinstance(itemset, tuple):
        return '{' + ', '.join(itemset) + '}'
    elif isinstance(itemset, frozenset):
        return '{' + ', '.join(sorted(list(itemset))) + '}'
    else:
        return str(itemset)


def print_brute_force_results(frequent_itemsets, rules, total_transc):
    print("\n=== Brute Force Algorithm Results ===")
    
    print("\nFrequent Itemsets:")
    print("-" * 140)
    print(f"{'Itemset':<120} {'Count':<10} {'Support (%)':<12}")
    print("-" * 140)
    
    sorted_itemsets = sorted(frequent_itemsets.items(), key=lambda x: (-x[1], len(x[0])))
    
    for itemset, support in sorted_itemsets:
        count = int(support * total_transc)  # Convert support to count
        print(f"{format_itemset(itemset):<120} {count:<10} {support * 100:.2f}%")
    
    print("\nAssociation Rules:")
    print("-" * 140)
    print(f"{'Rule':<120} {'Support (%)':<12} {'Confidence (%)':<12}")
    print("-" * 140)
    
    sorted_rules = sorted(rules, key=lambda x: -x['confidence'])
    
    for rule in sorted_rules:
        rule_str = f"{format_itemset(rule['antecedent'])} => {format_itemset(rule['consequent'])}"
        print(f"{rule_str:<120} {rule['support'] * 100:.2f}% \t {rule['confidence'] * 100:.2f}%")


def print_apriori_results(frequent_itemsets, rules, total_transc):
    print("\n=== Apriori Algorithm Results (mlxtend) ===")
    
    print(f"Total Frequent Itemsets Found: {len(frequent_itemsets)}")

    print("\nFrequent Itemsets:")
    print("-" * 140)
    print(f"{'Itemset':<120} {'Count':<10} {'Support (%)':<12}")
    print("-" * 140)
    
    for _, row in frequent_itemsets.iterrows():
        support = row['support']
        count = int(support * total_transc)  # Convert support to count
        itemset = row['itemsets']
        print(f"{format_itemset(itemset):<120} {count:<10} {support * 100:.2f}%")

    print(f"\nTotal Association Rules Generated: {len(rules)}")

    print("\nAssociation Rules:")
    print("-" * 140)
    print(f"{'Rule':<120} {'Support (%)':<12} {'Confidence (%)':<12}")
    print("-" * 140)
    
    for _, row in rules.iterrows():
        rule_str = f"{format_itemset(row['antecedents'])} => {format_itemset(row['consequents'])}"
        print(f"{rule_str:<120} {row['support'] * 100:.2f}% \t {row['confidence'] * 100:.2f}%")

# Main Function
# ------------
def main():
  
    print("=" * 60)
    print("Association Rule Mining - Comparison of Algorithms")
    print("=" * 60)
    print("\nAvailable datasets:")
    print("1. Amazon")
    print("2. K-mart")
    print("3. Best Buy")
    print("4. Nike")
    print("5. Walmart")
    
    # user input
    while True:
        dataset_choice = input("\nSelect a dataset (1-5): ")
        if dataset_choice in ['1', '2', '3', '4', '5']:
            break
        print("******Invalid choice! Please enter a number between 1 and 5.")

    while True:
        try:
            min_support = float(input("\nEnter minimum support threshold (0-1): "))
            if 0.0 <= min_support <= 1.0:
                break
            else:
                print("******Invalid input! Support must be between 0 and 1.")
        except ValueError:
            print("******Invalid input! Please enter a decimal number between 0 and 1.")

    while True:
        try:
            min_confidence = float(input("\nEnter minimum confidence threshold (0-1): "))
            if 0.0 <= min_confidence <= 1.0:
                break
            else:
                print("Invalid input! Confidence must be between 0 and 1.")
        except ValueError:
            print(" ******Invalid input! Please enter a decimal number between 0 and 1.")

    
    if dataset_choice == '1':
        dataset_name = "Amazon"
        items_file = "./dataset/amazon_items.csv"
        transactions_file = "./dataset/amazon_transactions.csv"
    elif dataset_choice == '2':
        dataset_name = "K-mart"
        items_file = "./dataset/kmart_items.csv"
        transactions_file = "./dataset/kmart_transactions.csv"
    elif dataset_choice == '3':
        dataset_name = "Best Buy"
        items_file = "./dataset/bestbuy_items.csv"
        transactions_file = "./dataset/bestbuy_transactions.csv"
    elif dataset_choice == '4':
        dataset_name = "Nike"
        items_file = "./dataset/nike_items.csv"
        transactions_file = "./dataset/nike_transactions.csv"
    elif dataset_choice == '5':
        dataset_name = "Walmart"
        items_file = "./dataset/walmart_items.csv"
        transactions_file = "./dataset/walmart_transactions.csv"
    else:
        print("Invalid choice. Exiting.")
        return
    
    # Checking if the file exist or not
    if not os.path.exists(items_file) or not os.path.exists(transactions_file):
        print(f"Error: Cannot find data files for {dataset_name}.")
        return
    
    print("=" * 60)
    print(f"\nAnalyzing {dataset_name} dataset with min_support={min_support} and min_confidence={min_confidence}")
    
    items_dict, transactions = read_data(items_file, transactions_file)
    print(f"Loaded {len(items_dict)} items and {len(transactions)} transactions.")
    one_hot_df = create_one_hot_encoding(transactions, items_dict) ## for mlextend apriori
    
    
    print("\nBrute Force algorithm:")
    frequent_itemsets_bf, bf_frequent_time = brute_force_frequent_itemsets(transactions, min_support)
    rules_bf, bf_rules_time = generate_association_rules_brute_force(frequent_itemsets_bf, min_confidence)
    bf_total_time = bf_frequent_time + bf_rules_time
    
    print("Running Apriori algorithm (mlxtend):")
    try:
        frequent_itemsets_ap, rules_ap, ap_total_time, ap_frequent_time,ap_rules_time  = run_apriori_mlxtend(one_hot_df, min_support, min_confidence)
    except:
        print("Couldn't find required association with given support and confidence")
        exit(1)    
    print_brute_force_results(frequent_itemsets_bf, rules_bf,len(transactions))
    print_apriori_results(frequent_itemsets_ap, rules_ap,len(transactions))
    
    print("\n=== Performance Comparison ===")
    print(f"Brute Force Total Time: {bf_total_time:.6f} seconds")
    print(f"  - Frequent Itemsets: {bf_frequent_time:.6f} seconds")
    print(f"  - Association Rules: {bf_rules_time:.6f} seconds")
    print(f"Apriori Total Time:    {ap_total_time:.6f} seconds")
    print(f"  - Frequent Itemsets: {ap_frequent_time:.6f} seconds")
    print(f"  - Association Rules: {ap_rules_time:.6f} seconds")
    
    if bf_total_time < ap_total_time:
        print(f"Brute Force was {ap_total_time/bf_total_time:.2f}x faster!")
    else:
        print(f"Apriori was {bf_total_time/ap_total_time:.2f}x faster!")
    
    bf_frequent_count = len(frequent_itemsets_bf)
    ap_frequent_count = len(frequent_itemsets_ap)
    bf_rules_count = len(rules_bf)
    ap_rules_count = len(rules_ap)
    
    print("\n=== Results Comparison ===")
    print(f"Brute Force: {bf_frequent_count} frequent itemsets, {bf_rules_count} rules")
    print(f"Apriori:     {ap_frequent_count} frequent itemsets, {ap_rules_count} rules")
    
    if bf_frequent_count == ap_frequent_count and bf_rules_count == ap_rules_count:
        print("Both algorithms produced the same number of results!")
    else:
        print("Results differ between the algorithms.")
        print("This might be due to implementation details or handling of edge cases.")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    main()
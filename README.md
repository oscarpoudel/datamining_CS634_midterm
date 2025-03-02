# Association Rule Mining: Brute Force vs. Apriori

## Project Overview
This project implements **Association Rule Mining** using two methods:
1. **Brute Force Algorithm** - Exhaustively generates all possible frequent itemsets and rules.
2. **Apriori Algorithm (mlxtend)** - An optimized method that prunes infrequent itemsets early.

The program allows users to analyze transaction data from different datasets, extract frequent itemsets, generate association rules, and compare execution times between the two approaches.

---

##  Features
 Supports **five retail datasets**: Amazon, K-Mart, Best Buy, Nike, and Walmart.  
 **User-specified** support and confidence thresholds.  
 **Brute Force Algorithm** for exhaustive rule generation.  
 **Apriori Algorithm (mlxtend)** for efficient rule extraction.  
 **Performance comparison** between both methods.  
 **Formatted results** with support (%) and confidence (%).  
 **Error handling** for invalid user input.  

---

##  Installation
###  Clone the Repository
```bash
git clone https://github.com/yourusername/association-rule-mining.git
cd association-rule-mining

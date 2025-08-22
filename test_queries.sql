# ============================================================================
# FILE: test_queries.sql
# ============================================================================
-- USER EDIT: Add your test queries here
-- Each query should be separated by a semicolon
-- Use table names that match your data

-- Query 1: Simple select with filter
SELECT * FROM table1 WHERE value > 100;

-- Query 2: Basic join
SELECT t1.id, t1.value, t2.category
FROM table1 t1
JOIN table2 t2 ON t1.id = t2.id;

-- Query 3: Aggregation with groupby
SELECT category, COUNT(*), AVG(value)
FROM table1
GROUP BY category;

-- Query 4: Multi-way join
SELECT t1.id, t2.category, t3.description
FROM table1 t1
JOIN table2 t2 ON t1.id = t2.id
JOIN table3 t3 ON t2.category_id = t3.id
WHERE t1.value > 50;

-- Query 5: Complex aggregation
SELECT 
    t1.category,
    COUNT(DISTINCT t1.id) as unique_ids,
    SUM(t2.amount) as total_amount
FROM table1 t1
LEFT JOIN table2 t2 ON t1.id = t2.foreign_id
GROUP BY t1.category
HAVING COUNT(DISTINCT t1.id) > 10;
-- Feature view for the MySQL "employees" sample database
-- (tables: employees, departments, dept_emp, dept_manager, titles, salaries).
-- One row per employee, ready for machine learning.
--
-- Run ONCE in MySQL Workbench as root (connected to your 3305 server), then:
--     python train_models.py --config employees_training_config.yaml
--
-- The employees data ends in 2002, and rows still valid today have to_date = '9999-01-01'.

USE employees;

DROP VIEW IF EXISTS employee_features;

CREATE VIEW employee_features AS
SELECT
    e.emp_no,
    e.gender,
    TIMESTAMPDIFF(YEAR, e.birth_date, e.hire_date)            AS age_at_hire,
    YEAR(e.hire_date)                                         AS hire_year,
    d.dept_name                                               AS department,
    t.title                                                   AS current_title,
    t.n_titles,
    (dm.emp_no IS NOT NULL)                                   AS is_manager,
    s.salary_records,
    s.min_salary,
    s.max_salary,
    ROUND((s.max_salary - s.min_salary) / s.min_salary * 100, 1) AS salary_growth_pct,
    TIMESTAMPDIFF(YEAR, e.hire_date, LEAST(s.last_to_date, '2002-08-01')) AS years_of_service,
    CASE WHEN s.last_to_date = '9999-01-01' THEN 0 ELSE 1 END AS left_company
FROM employees e
JOIN (
    SELECT emp_no, COUNT(*) AS salary_records, MIN(salary) AS min_salary,
           MAX(salary) AS max_salary, MAX(to_date) AS last_to_date
    FROM salaries GROUP BY emp_no
) s ON s.emp_no = e.emp_no
JOIN (
    SELECT emp_no, dept_no
    FROM (SELECT emp_no, dept_no,
                 ROW_NUMBER() OVER (PARTITION BY emp_no ORDER BY to_date DESC, from_date DESC) AS rn
          FROM dept_emp) x
    WHERE rn = 1
) de ON de.emp_no = e.emp_no
JOIN departments d ON d.dept_no = de.dept_no
JOIN (
    SELECT emp_no, title, n_titles
    FROM (SELECT emp_no, title,
                 COUNT(*) OVER (PARTITION BY emp_no) AS n_titles,
                 ROW_NUMBER() OVER (PARTITION BY emp_no ORDER BY to_date DESC, from_date DESC) AS rn
          FROM titles) y
    WHERE rn = 1
) t ON t.emp_no = e.emp_no
LEFT JOIN (SELECT DISTINCT emp_no FROM dept_manager) dm ON dm.emp_no = e.emp_no;

-- quick check
SELECT left_company, COUNT(*) AS employees, ROUND(AVG(max_salary)) AS avg_max_salary
FROM employee_features GROUP BY left_company;

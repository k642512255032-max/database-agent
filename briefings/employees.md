# Database briefing: employees (MySQL sample database)

## What it is
The classic MySQL "employees" sample database: HR records of a fictional company, ~300,000 employees, with the
full history of their departments, titles and salaries from 1985 to 2002. It is a **snapshot that ends in
2002-08-01**; nothing changes after that date.

## Tables (grain, keys, size)
- **employees** (1 row per employee, PK emp_no; ~300,024 rows): birth_date, first_name, last_name, gender (M/F),
  hire_date. Birth dates 1952-1965, hire dates 1985-2000.
- **departments** (1 row per department, PK dept_no; 9 rows d001-d009): dept_name (Marketing, Finance, Human
  Resources, Production, Development, Quality Management, Sales, Research, Customer Service).
- **dept_emp** (1 row per employee *per department assignment*, PK (emp_no, dept_no); ~331,000 rows): from_date,
  to_date. An employee can appear twice (moved department). Current assignment: to_date = '9999-01-01'.
- **dept_manager** (1 row per manager assignment, PK (emp_no, dept_no); 24 rows): from_date, to_date. Managers are
  ordinary employees who also appear here. Current manager: to_date = '9999-01-01'.
- **salaries** (1 row per employee *per salary period*, PK (emp_no, from_date); ~2.84M rows): salary, from_date,
  to_date. Roughly one row per year per employee (annual raise). Current salary = the row with
  to_date = '9999-01-01'; employees who left have no current row. Salaries range about 38,600 - 158,000.
- **titles** (1 row per employee *per title period*, PK (emp_no, title, from_date); ~443,000 rows): title
  (Staff, Senior Staff, Engineer, Senior Engineer, Assistant Engineer, Technique Leader, Manager), from_date,
  to_date. Current title: to_date = '9999-01-01'.
- **current_dept_emp** (view): emp_no, dept_no, from_date, to_date - each employee's latest department assignment.
- **dept_emp_latest_date** (view): emp_no with the latest from_date / to_date in dept_emp.
- **employee_features** (view or materialised table, 1 row per employee, for ML): emp_no, gender, age_at_hire,
  hire_year, department, current_title, n_titles, is_manager, salary_records, min_salary, max_salary,
  salary_growth_pct, years_of_service (to 2002-08-01), left_company (1 if the last salary row ended before
  9999-01-01).

## Joins
employees.emp_no = dept_emp.emp_no = salaries.emp_no = titles.emp_no = dept_manager.emp_no;
dept_emp.dept_no = departments.dept_no = dept_manager.dept_no.

## Rules of thumb an expert applies
- "Current" anything = to_date = '9999-01-01'. Without that filter, joins to salaries / titles / dept_emp multiply
  rows (one employee x many history rows) and inflate counts, sums and averages.
- "Employees who left" = no salary row with to_date = '9999-01-01' (or employee_features.left_company = 1).
  About 240,000 of 300,000 employees still have a current row.
- "Salary by department" is ambiguous: current salary of current department members (usual meaning) vs every
  salary ever paid. Say which one the query uses.
- Tenure / years of service: the data stops in 2002, so measuring to CURRENT_DATE gives 25+ years for everyone;
  measure to the snapshot date '2002-08-01' (as employee_features does) unless the user explicitly wants "as of today".
- Age: birth_date is 1952-1965, so ages "today" are 60+; ages at hire or in 2002 are more meaningful.
- dept_manager has only 24 rows: manager questions are about those 24 people, not about titles = 'Manager'.
- There are no NULLs in the base tables; suspicious values are more likely sentinel dates than missing data.
- Names are not unique (many employees share first_name + last_name); identify people by emp_no.
- Gender split is roughly 60% M / 40% F; pay-gap questions should compare current salaries within the same
  title or department, not the overall average.

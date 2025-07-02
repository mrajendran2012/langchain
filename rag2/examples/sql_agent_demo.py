import sqlite3

def create_demo_db():
    conn = sqlite3.connect(":memory:")
    cursor = conn.cursor()
    cursor.execute("CREATE TABLE employees (id INTEGER PRIMARY KEY, name TEXT, department TEXT, salary INTEGER)")
    cursor.executemany(
        "INSERT INTO employees (name, department, salary) VALUES (?, ?, ?)",
        [
            ("Alice", "Engineering", 120000),
            ("Bob", "Sales", 90000),
            ("Charlie", "HR", 70000),
            ("Diana", "Engineering", 115000),
        ]
    )
    conn.commit()
    return conn

def ask_sql_agent(question, conn):
    # Simple demo: map question to SQL query (in real use, use LLM or langchain agent)
    if "highest salary" in question.lower():
        query = "SELECT name, salary FROM employees ORDER BY salary DESC LIMIT 1"
    elif "engineering" in question.lower():
        query = "SELECT name FROM employees WHERE department = 'Engineering'"
    elif "average salary" in question.lower():
        query = "SELECT AVG(salary) FROM employees"
    else:
        return "Sorry, I can't answer that."
    cursor = conn.cursor()
    cursor.execute(query)
    result = cursor.fetchall()
    return result

if __name__ == "__main__":
    conn = create_demo_db()
    print("Ask a question about the employees database (e.g., 'Who has the highest salary?')\nType 'exit' to quit.")
    while True:
        question = input("Your question: ")
        if question.lower() == "exit":
            break
        answer = ask_sql_agent(question, conn)
        print("Answer:", answer)
    conn.close()
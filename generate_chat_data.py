import json
import random

def generate_dataset(filepath="chat_dataset.jsonl"):
    data = []
    
    # 1. Python Examples
    data.append({"prompt": "How do I print hello world in Python?", "response": "In Python, you simply use the built-in `print` function:\n\n```python\nprint(\"Hello, World!\")\n```"})
    data.append({"prompt": "How do I define a function in Python?", "response": "You use the `def` keyword followed by the function name and parameters:\n\n```python\ndef greet(name):\n    return f\"Hello, {name}!\"\n```"})
    
    # 2. Java Examples
    data.append({"prompt": "How do I print something in Java?", "response": "In Java, you use the `System.out.println` method.\n\n```java\nSystem.out.println(\"Hello, World!\");\n```"})
    data.append({"prompt": "Write a Java class that says hello.", "response": "Certainly! Here is a minimal Java program:\n\n```java\npublic class Main {\n    public static void main(String[] args) {\n        System.out.println(\"Hello, World!\");\n    }\n}\n```"})
    
    # 3. JavaScript Examples
    data.append({"prompt": "What is a React hook in JavaScript?", "response": "A React hook is a special function that lets you \"hook into\" React features like state and lifecycle methods. The most common one is `useState`.\n\n```javascript\nimport { useState } from 'react';\n\nfunction Counter() {\n    const [count, setCount] = useState(0);\n    return <button onClick={() => setCount(count + 1)}>{count}</button>;\n}\n```"})
    data.append({"prompt": "How do I handle async operations in JS?", "response": "The modern approach is to use `async` and `await` keywords which make asynchronous code look synchronous.\n\n```javascript\nasync function fetchData() {\n    const response = await fetch('https://api.example.com/data');\n    const data = await response.json();\n    console.log(data);\n}\n```"})
    
    # Duplicate and rephrase variations to reach ~75 examples for robust mechanical training
    templates = [
        ("I need to {}.", "To {}, you can use this approach:\n\n"),
        ("Show me how to {}.", "Here is an example of how to {}:\n\n"),
        ("Can you explain {}?", "Certainly! {} is straightforward:\n\n"),
        ("What's the syntax for {}?", "The syntax for {} looks like this:\n\n")
    ]
    
    topics = [
        ("declare a constant in JavaScript", "use `const`:\n`const PI = 3.14;`"),
        ("create a list in Python", "Use square brackets:\n`my_list = [1, 2, 3]`"),
        ("use an ArrayList in Java", "`List<Integer> numbers = new ArrayList<>();`"),
        ("write a while loop in Python", "`while i < 10:\n    i += 1`"),
        ("map an array in JS", "`const doubled = arr.map(x => x * 2);`"),
        ("import a class in Java", "Use the `import` keyword:\n`import java.util.List;`"),
    ]
    
    for _ in range(60):
        t_q, t_a = random.choice(templates)
        topic, answer = random.choice(topics)
        q = t_q.format(topic)
        a = t_a.format(topic) + answer
        data.append({"prompt": q, "response": a})
        
    random.shuffle(data)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(data)} examples and saved to {filepath}")

if __name__ == "__main__":
    generate_dataset()

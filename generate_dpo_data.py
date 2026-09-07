import json
import random

def generate_dataset(filepath="dpo_dataset.jsonl"):
    data = []
    
    templates = [
        ("I need to {}.", "To {}, you can use this approach:\n\n", "You can {} by writing some code for it."),
        ("Show me how to {}.", "Here is an example of how to {}:\n\n", "Just look up how to {} in the documentation."),
        ("Can you explain {}?", "Certainly! {} is straightforward:\n\n", "{} is a programming feature."),
        ("What's the syntax for {}?", "The syntax for {} looks like this:\n\n", "The syntax for {} is pretty standard.")
    ]
    
    topics = [
        ("declare a constant in JavaScript", "use `const`:\n`const PI = 3.14;`"),
        ("create a list in Python", "Use square brackets:\n`my_list = [1, 2, 3]`"),
        ("use an ArrayList in Java", "`List<Integer> numbers = new ArrayList<>();`"),
        ("write a while loop in Python", "`while i < 10:\n    i += 1`"),
        ("map an array in JS", "`const doubled = arr.map(x => x * 2);`"),
        ("import a class in Java", "Use the `import` keyword:\n`import java.util.List;`"),
        ("print hello world in Python", "Use print:\n`print(\"Hello, World!\")`"),
        ("allocate memory in Java", "Use the new keyword:\n`MyClass obj = new MyClass();`"),
        ("use promises in JS", "Use async/await:\n`const data = await fetch(url);`"),
        ("write a class in Python", "Use the class keyword:\n`class MyClass:\n    pass`"),
        ("define a function in Java", "Specify return type:\n`public int add(int a, int b) { return a + b; }`")
    ]
    
    for _ in range(75):
        t_q, t_chosen_prefix, t_rejected = random.choice(templates)
        topic, answer = random.choice(topics)
        
        q = t_q.format(topic)
        chosen = t_chosen_prefix.format(topic) + answer
        rejected = t_rejected.format(topic)
        
        data.append({
            "prompt": q,
            "chosen": chosen,
            "rejected": rejected
        })
        
    random.shuffle(data)
    
    with open(filepath, "w", encoding="utf-8") as f:
        for item in data:
            f.write(json.dumps(item) + "\n")
            
    print(f"Generated {len(data)} DPO examples and saved to {filepath}")

if __name__ == "__main__":
    generate_dataset()

from collections import defaultdict

def load_graph(filename):
    rows = []  
    cols = []  
    values = []  
    out_degree = defaultdict(int)
    nodes = set()  
    node_to_id = {}

    with open(filename, 'r') as file:
        for line in file:
            if line.startswith("#"):
                continue
            from_node, to_node = map(int, line.split())
            
            if from_node not in node_to_id:
                node_to_id[from_node] = len(node_to_id)
            if to_node not in node_to_id:
                node_to_id[to_node] = len(node_to_id)

            from_idx = node_to_id[from_node]
            to_idx = node_to_id[to_node]
            
            rows.append(to_idx)
            cols.append(from_idx)
            values.append(1)  
            out_degree[from_node] += 1
            nodes.add(from_node)
            nodes.add(to_node)

    num_nodes = len(node_to_id)
    row_pointers = [0] * (num_nodes + 1)

    for i in range(len(rows)):
        row_pointers[rows[i] + 1] += 1
    
    for i in range(1, len(row_pointers)):
        row_pointers[i] += row_pointers[i - 1]

    return row_pointers, cols, values, out_degree, node_to_id

def power_iteration_csr(row_pointers, cols, values, out_degree, nodes, num_iterations=100, epsilon=1e-6):
    num_nodes = len(nodes)
    r = {i: 1 / num_nodes for i in range(num_nodes)}  
    jump_factor = 0.85

    for _ in range(num_iterations):
        new_r = {i: (1 - jump_factor) / num_nodes for i in range(num_nodes)}

        for i in range(num_nodes):
            row_start = row_pointers[i]
            row_end = row_pointers[i + 1]
            for j in range(row_start, row_end):
                in_node = cols[j]
                if out_degree.get(in_node, 0) > 0:
                    new_r[i] += jump_factor * r[in_node] / out_degree.get(in_node, 1)

        diff = sum(abs(new_r[i] - r[i]) for i in range(num_nodes))
        
        # (|new_r[i] - r[i]|)의 합이 epsilon보다 작아지면 종료
        if diff < epsilon:
            break

        r = new_r

    return r

def main():
    filename = 'web-Google.txt'  

    row_pointers, cols, values, out_degree, node_to_id = load_graph(filename)

    importance = power_iteration_csr(row_pointers, cols, values, out_degree, node_to_id)

    sorted_importance = sorted(node_to_id.items(), key=lambda x: importance[x[1]], reverse=True)
    
    with open('poweriteration_output.txt', 'w') as f:
        for node, index in sorted_importance:
            f.write(f"{node}\t{importance[index]:.6f}\n")

if __name__ == "__main__":
    main()

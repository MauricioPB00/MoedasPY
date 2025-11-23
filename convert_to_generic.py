import os

def convert_label_file(path):
    with open(path, "r") as f:
        lines = f.readlines()

    new_lines = []
    for line in lines:
        parts = line.strip().split()
        if len(parts) != 5:
            continue

        parts[0] = "0"

        new_lines.append(" ".join(parts) + "\n")

    with open(path, "w") as f:
        f.writelines(new_lines)


def convert_all_labels(labels_dir):
    count = 0
    for root, dirs, files in os.walk(labels_dir):
        for file in files:
            if file.endswith(".txt"):
                convert_label_file(os.path.join(root, file))
                count += 1

    print(f"Conversão concluída! {count} arquivos convertidos.")


if __name__ == "__main__":
    #labels_folder = r"C:\Users\Usuario\Documents\Moedas\MoedasPY\dataset\labels\val"  # <-- ajuste aqui
    labels_folder = r"C:\Users\Usuario\Documents\Moedas\MoedasPY\dataset\labels\train"  # <-- ajuste aqui
    convert_all_labels(labels_folder)

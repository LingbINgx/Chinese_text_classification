import csv

filename = ["cnews.train.txt", "cnews.val.txt", "cnews.test.txt"]

for name in filename:
    with open(f"../data/{name}", "r", encoding="utf-8") as f:
        rows = []
        for line in f:
            parts = line.strip().split("\t", 1)
            if len(parts) == 2:
                label, content = parts
                rows.append([label, content])

        # 写入 CSV 文件
        output_file = f"../data/{name.replace('.txt', '.csv')}"
        with open(output_file, "w", newline="", encoding="utf-8-sig") as fp:
            writer = csv.writer(fp)
            writer.writerow(["label", "content"])
            writer.writerows(rows)

        print(f"CSV 文件已生成：{output_file}")
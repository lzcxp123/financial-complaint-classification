"""
金融消费者投诉文本分类系统 - 前端统一启动脚本
用法:
  python run.py                    # 交互式菜单
  python run.py --all              # 同时启动所有前端
  python run.py --model textcnn    # 启动指定模型前端
"""
import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

MODELS = [
    {
        "key": "textcnn",
        "name": "TextCNN",
        "path": "02-textcnn/app.py",
        "port": 8501,
        "api_port": 8002,
    },
    {
        "key": "bilstm",
        "name": "BiLSTM+Attention",
        "path": "03-bilstm/app.py",
        "port": 8502,
        "api_port": 8003,
    },
    {
        "key": "finbert",
        "name": "FinBERT",
        "path": "04-bert/src/app.py",
        "port": 8503,
        "api_port": 8004,
    },
    {
        "key": "distill",
        "name": "知识蒸馏",
        "path": "05-distill/src/app.py",
        "port": 8504,
        "api_port": 8005,
    },
]


def start_frontend(model_info):
    """启动单个模型的Streamlit前端"""
    cmd = [
        sys.executable, "-m", "streamlit", "run",
        model_info["path"],
        "--server.port", str(model_info["port"]),
        "--server.headless", "true",
    ]
    print(f"  启动 {model_info['name']} 前端 → http://localhost:{model_info['port']}")
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    return proc


def print_header():
    print("=" * 55)
    print("  金融消费者投诉文本分类系统 - 前端启动")
    print("=" * 55)
    for i, m in enumerate(MODELS, 1):
        print(f"  [{i}] {m['name']:<18} → http://localhost:{m['port']}")
    print(f"  [5] 全部启动")
    print(f"  [0] 退出")
    print("=" * 55)
    print("  注意: 请确保已先启动对应的Flask API服务 (端口8002-8005)")
    print("=" * 55)


def interactive_menu():
    """交互式菜单模式"""
    processes = []
    while True:
        print_header()
        choice = input("请选择要启动的前端 [0-5]: ").strip()

        if choice == "0":
            if processes:
                print("正在关闭所有前端进程...")
                for p in processes:
                    p.terminate()
            print("退出。")
            break

        if choice == "5":
            print("\n正在启动所有前端...")
            for m in MODELS:
                p = start_frontend(m)
                processes.append(p)
            print(f"\n✅ 已启动 {len(MODELS)} 个前端")
            print("按 Ctrl+C 停止所有进程")
            try:
                for p in processes:
                    p.wait()
            except KeyboardInterrupt:
                print("\n正在停止所有前端...")
                for p in processes:
                    p.terminate()
            break

        try:
            idx = int(choice) - 1
            if 0 <= idx < len(MODELS):
                m = MODELS[idx]
                print(f"\n正在启动 {m['name']} 前端...")
                p = start_frontend(m)
                processes.append(p)
                print(f"✅ {m['name']} 已启动 → http://localhost:{m['port']}")
                input("\n按回车键返回菜单...")
            else:
                print("无效选择，请重新输入。")
        except ValueError:
            print("请输入数字 0-5。")


def main():
    args = sys.argv[1:]

    if "--all" in args:
        print_header()
        print("\n正在启动所有前端...")
        processes = []
        for m in MODELS:
            p = start_frontend(m)
            processes.append(p)
        print(f"\n✅ 已启动 {len(MODELS)} 个前端")
        print("按 Ctrl+C 停止所有进程")
        try:
            for p in processes:
                p.wait()
        except KeyboardInterrupt:
            print("\n正在停止所有前端...")
            for p in processes:
                p.terminate()
        return

    if "--model" in args:
        idx = args.index("--model")
        if idx + 1 < len(args):
            model_key = args[idx + 1].lower()
            model_map = {m["key"]: m for m in MODELS}
            if model_key in model_map:
                m = model_map[model_key]
                print(f"启动 {m['name']} 前端 → http://localhost:{m['port']}")
                p = start_frontend(m)
                p.wait()
            else:
                print(f"未知模型: {model_key}")
                print(f"可用模型: {', '.join(m['key'] for m in MODELS)}")
        else:
            print("请指定模型名称，如 --model textcnn")
        return

    interactive_menu()


if __name__ == "__main__":
    main()

import gradio as gr
import subprocess
import os
import time


def run_command():
    """
    执行外部命令的函数
    """
    # 构造命令路径
    config_path = "/tmp/pycharm_project_ATD2/options/test/003_ATD_SRx4_finetune.yml"

    # 检查文件是否存在
    if not os.path.exists(config_path):
        return f"错误：配置文件不存在 - {config_path}"

    try:
        # 执行命令
        result = subprocess.run(
            ["python", "-opt", config_path],
            capture_output=True,
            text=True,
            check=True
        )

        # 返回命令输出
        output = f"命令执行成功！\n\n输出:\n{result.stdout}"
        if result.stderr:
            output += f"\n\n错误输出:\n{result.stderr}"
        return output

    except subprocess.CalledProcessError as e:
        # 处理命令执行错误
        return f"命令执行失败！\n\n返回码: {e.returncode}\n输出:\n{e.stdout}\n错误:\n{e.stderr}"
    except Exception as e:
        # 处理其他异常
        return f"发生意外错误: {str(e)}"


# 创建Gradio界面
demo = gr.Interface(
    fn=run_command,
    inputs=[],
    outputs="text",
    title="执行配置文件命令",
    description="点击按钮执行命令: python -opt /tmp/pycharm_project_ATD2/options/test/003_ATD_SRx4_finetune.yml"
)

# 启动应用
if __name__ == "__main__":
    demo.launch(share=True, server_port=4545)
---
alwaysApply: false
globs: '**/*.py,**/*.pyi'
---

# AI 生成 Python 代码的要求

> 通用原则（工作流、编码偏好、完成验证、审核指南）见 `../std-common/ai-workflow.mdc`。本文件仅补充 Python 语言特有约束。

---

## 类型提示

所有函数/方法**必须**有完整的参数和返回值类型注解，优先使用 Python 3.10+ 语法（`X | None` 代替 `Optional[X]`）。

---

## 错误处理

- **禁止裸 `except` 和 `except Exception`**：必须捕获具体异常类型
- **异常链必须保留**：`raise NewError(...) from e`
- **资源必须用 `with` 管理**：文件、连接、锁等

```python
# ✅
try:
    data = json.loads(raw)
except json.JSONDecodeError as e:
    raise ValidationError("JSON 格式错误") from e

# ❌
try:
    data = json.loads(raw)
except:
    pass
```

---

## 可变默认参数（红线）

Python 经典陷阱——可变对象作为默认参数会在调用间共享状态：

```python
# ❌ 所有调用共享同一个 list
def append_to(element: str, target: list[str] = []) -> list[str]: ...

# ✅ 使用 None 哨兵
def append_to(element: str, target: list[str] | None = None) -> list[str]:
    if target is None:
        target = []
    target.append(element)
    return target
```

---

## 异步代码

- `async def` 内的所有异步调用**必须 `await`**，遗漏 await 会导致协程静默不执行
- 异步上下文管理器用 `async with` 确保资源释放

---

## Python 特有审核清单

生成 Python 代码后，在通用审核指南基础上额外检查：

- [ ] 函数/方法是否有完整类型注解
- [ ] 是否存在裸 `except` 或 `except Exception`
- [ ] 是否存在可变默认参数
- [ ] 异步代码是否正确 `await`
- [ ] 资源（文件/连接）是否使用 `with` 管理
- [ ] 异常链是否保留（`from e`）

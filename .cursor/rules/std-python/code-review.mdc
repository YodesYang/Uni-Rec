---
description: Python Code Review 规范 - 当用户请求对 Python 代码进行代码评审时自动应用
alwaysApply: false
---

# Python Code Review 规范

> 融合 PEP 8、Google Python Style Guide 与 mypy 最佳实践

---

## 评审关注点

### P0：正确性与安全（必须通过）

**业务正确性**
- [ ] 需求实现完整性
- [ ] 边界条件处理（None、空集合、边界值）
- [ ] 异常处理完整性

**安全性**
- [ ] 无 SQL 注入（使用参数化查询）
- [ ] 无命令注入
- [ ] 敏感信息不硬编码
- [ ] 输入验证完整

### P1：类型安全与错误处理（高优先级）

**类型安全**
- [ ] 公开函数有完整类型注解
- [ ] 使用 Optional 而非 None 联合类型
- [ ] mypy strict 模式通过

**错误处理**
- [ ] 无裸 except
- [ ] 异常类型具体
- [ ] 使用异常链 `from e`
- [ ] 资源使用 with 语句管理

```python
# ✅ 推荐：具体异常 + 异常链
try:
    data = json.loads(raw_data)
except json.JSONDecodeError as e:
    raise ValidationError(f"JSON 解析失败: {e}") from e

# ❌ 避免：裸 except
try:
    data = json.loads(raw_data)
except:
    data = {}  # 吞掉所有异常！
```

### P2：默认参数（高优先级）

- [ ] 避免可变默认参数
- [ ] 使用 None 作为可变类型的默认值

```python
# ❌ 危险：可变默认参数
def append_item(item: str, items: list[str] = []) -> list[str]:
    items.append(item)  # 多次调用会累积！
    return items

# ✅ 推荐：使用 None
def append_item(item: str, items: list[str] | None = None) -> list[str]:
    if items is None:
        items = []
    items.append(item)
    return items
```

### P3：异步代码（高优先级，如适用）

- [ ] 正确 await 协程
- [ ] 使用 async with 管理异步资源
- [ ] 避免在异步函数中调用阻塞 I/O
- [ ] 正确使用 asyncio.gather/TaskGroup

```python
# ✅ 推荐
async def fetch_all(urls: list[str]) -> list[Response]:
    async with httpx.AsyncClient() as client:
        tasks = [client.get(url) for url in urls]
        return await asyncio.gather(*tasks)

# ❌ 避免：忘记 await
async def fetch_user(client: httpx.AsyncClient, user_id: int) -> User:
    response = client.get(f"/users/{user_id}")  # 忘记 await！
    return User(**response.json())

# ❌ 避免：异步函数中调用阻塞 I/O
async def read_file(path: str) -> str:
    with open(path) as f:  # 阻塞操作！
        return f.read()

# ✅ 推荐：使用异步 I/O
async def read_file(path: str) -> str:
    async with aiofiles.open(path) as f:
        return await f.read()
```

### P4：代码风格（中优先级）

- [ ] 遵循 PEP 8 命名规范
- [ ] 导入顺序正确（标准库 → 第三方 → 本地）
- [ ] 使用 f-string 而非 % 或 format
- [ ] 使用列表推导式而非 map/filter

```python
# ✅ Pythonic 风格
names = [user.name for user in users if user.is_active]
message = f"Hello, {name}! You have {count} messages."

# ❌ 非 Pythonic 风格
names = list(map(lambda u: u.name, filter(lambda u: u.is_active, users)))
message = "Hello, %s! You have %d messages." % (name, count)
```

### P5：文档与测试（中优先级）

- [ ] 公开函数有 docstring
- [ ] docstring 包含 Args/Returns/Raises
- [ ] 有对应的单元测试
- [ ] 测试覆盖边界情况

### P6：性能（按需）

- [ ] 大数据集使用生成器
- [ ] 避免不必要的列表复制
- [ ] 使用集合/字典进行查找
- [ ] 考虑缓存（lru_cache）

### P7：依赖与工具链（低优先级）

**AI 评审前必须识别项目实际工具**：
- Python 版本：检查 `pyproject.toml` 或 `setup.py`
- 依赖管理：pip/poetry/pdm/uv
- 类型检查：mypy/pyright
- Linter：ruff/flake8/pylint
- 格式化：black/ruff

---

## 参考资源

- [PEP 8 - Style Guide for Python Code](https://peps.python.org/pep-0008/)
- [PEP 484 - Type Hints](https://peps.python.org/pep-0484/)
- [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html)
- [mypy Documentation](https://mypy.readthedocs.io/)
- [Real Python](https://realpython.com/)


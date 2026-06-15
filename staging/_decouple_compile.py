"""Stage-2 compile decoupling — self-verifying editor.

Runs entirely in-process (no reliance on terminal echo). Each file is edited
in memory; ALL preconditions are asserted before ANY file is written, so a
wrong anchor aborts with a distinct exit code and leaves the tree untouched.

Exit codes: 0 = all files edited OK. Non-zero = (file_index*10 + step) where
step: 1 import line, 2 init attr, 3 apply method, 4 forward dispatch,
5 residual refs. 90 = post-edit re-read mismatch.
"""

import re
import sys

ROOT = "openwam/model/architectures/"


def strip_apply_method(s):
    pat = re.compile(
        r"\n    def apply_compile_optimizations\(self, compile_cfg\) -> None:"
        r".*?(?=\n    def |\n    @|\n\nclass |\n__all__|\Z)",
        re.DOTALL,
    )
    return pat.subn("", s)


def do_joint_self_attn():
    p = ROOT + "dual_system/joint_self_attn.py"
    s = open(p).read()

    imp = "from openwam.model.architectures.dual_system.mot_compile import CompiledMoTLoop\n"
    if s.count(imp) != 1:
        return 1, None, None
    s = s.replace(imp, "")

    if len(re.findall(r"\n        self\._compiled_mot_loop[^\n]*= None", s)) != 1:
        return 2, None, None
    s = re.sub(r"\n        self\._compiled_mot_loop[^\n]*= None", "", s)

    s, n = strip_apply_method(s)
    if n != 1:
        return 3, None, None

    old_fwd = (
        "        compiled_loop = self._compiled_mot_loop\n"
        "        if compiled_loop is not None and compiled_loop.can_run(\n"
        "            vstate,\n"
        "            astate,\n"
        "            use_gradient_checkpointing=use_gradient_checkpointing,\n"
        "            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,\n"
        "        ):\n"
        "            vstate, astate = compiled_loop.run(vstate, astate)\n"
        "        else:\n"
        "            vstate, astate = driver.run_joint_loop(\n"
        "                vstate,\n"
        "                astate,\n"
        "                use_gradient_checkpointing=use_gradient_checkpointing,\n"
        "                use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,\n"
        "            )\n"
    )
    new_fwd = (
        "        vstate, astate = driver.run_joint_loop(\n"
        "            vstate,\n"
        "            astate,\n"
        "            use_gradient_checkpointing=use_gradient_checkpointing,\n"
        "            use_gradient_checkpointing_offload=use_gradient_checkpointing_offload,\n"
        "        )\n"
    )
    if s.count(old_fwd) != 1:
        return 4, None, None
    s = s.replace(old_fwd, new_fwd)

    for tok in ("_compiled_mot_loop", "CompiledMoTLoop", "apply_compile_optimizations", "compiled_loop"):
        if tok in s:
            return 5, None, None
    return 0, p, s


def main():
    results = []
    for idx, fn in enumerate([do_joint_self_attn], start=1):
        code, p, s = fn()
        if code != 0:
            sys.exit(idx * 10 + code)
        results.append((p, s))

    # All preconditions passed — write atomically.
    for p, s in results:
        open(p, "w").write(s)

    # Re-read to confirm writes landed and no residual compiled refs remain.
    for p, _ in results:
        s2 = open(p).read()
        if "_compiled_" in s2 and "self._compiled_" in s2:
            sys.exit(90)
    sys.exit(0)


if __name__ == "__main__":
    main()

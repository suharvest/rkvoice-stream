# Upstream bug report (draft) — RK1820/RK1828 RKNN3 V1.0.4: intermittent + degrading `MODEL_SETUP ACK_FAIL` at model init

> Draft to file with Rockchip (RKNN3 / RK182X SDK support). Consolidates the host-side evidence; the firmware-side root cause is not host-observable (see §5), so Rockchip input is required.

## Summary
On RK1828 (RK1820 NPU coprocessor) with **RKNN3 SDK V1.0.4**, `rknn3_model_init()` intermittently fails with `MODEL_SETUP fail, ack = ACK_FAIL` and the process aborts (SIGABRT, rc=134). Two regimes:
- **(A) cosmetic flaky** — the first load attempt fails, an in-process retry (2nd/3rd attempt) succeeds.
- **(B) degrading/persistent** — after some accumulated failures the EP enters a state where **every** subsequent `model_init` fails deterministically; the only recovery is a **clean firmware reflash** (host reboot → `rknn3.service` re-downloads firmware). Measured 11/11 consecutive failures, then 2/2 immediate success after one reboot.

The failure is at the **first** `rknn3_model_init` of a multi-model app (a Gemma-4 demo: an LLM model `core_mask=0xff` + an audio-encoder model `core_mask=0xf`); it aborts right after the tokenizer/metadata load, before any inference.

## Environment
- EP: RK1820/RK1828 (RM182X), PCIe2.1 x1 EP, `[1d87:182a]`, BDF `0001:11:00.0`, 5 GB.
- Host: RK3588 (Radxa ROCK 5T), kernel 6.1.x, `pcie-rkep` driver.
- SDK: **RKNN3 V1.0.4** (`rknn3_rk182x_m2_installer_arm64.tgz`).
- Firmware: `/lib/firmware/rknn3_rk1820.img`, **md5 `37ca5e4e1ac9fbb8479a042a35574760`** — byte-identical to the firmware shipped in the V1.0.4 installer (i.e. the latest available; not a stale older firmware).
- Link: gen2x1; `magic=524b4550 (RKEP)`.

## Exact error (host side)
```
E RKNNAPI(<pid>): MODEL_SETUP fail, ack = ACK_FAIL, expect = ACK_SUCC!
E RKNNAPI(<pid>): rknn3_model_init,  recv model setup fail ack!
terminate called without an active exception
→ SIGABRT (rc=134)
```
No `core_mask ... is not match with npu core number N` line accompanies it (the masks are correct — see §4).

## §3 — Suspected root cause: RC/EP firmware version skew (both from V1.0.4)
The PCIe handshake (from `rknn-smi` log / `cc_core_pcie` init) reports **mismatched component versions**, both originating from the V1.0.4 install:
```
rc_cc_version = 30301   (host RC side, v3.03.01)
ep_cc_version = 30201   (EP firmware side, v3.02.01)
```
i.e. the V1.0.4 host runtime (`librknnrt3` / `rknn3_transfer_proxy`) advertises **3.03.01** while the V1.0.4 EP firmware (`rknn3_rk1820.img`) reports **3.02.01**. We could not flash a matching EP firmware because the installed image is already the latest one shipped in V1.0.4.

**Q1: Is this RC↔EP version skew (30301 vs 30201) in V1.0.4 expected, or a packaging mismatch? Could it cause the intermittent model-setup ACK handshake failures?**

## §4 — What we ruled out (host side)
- **Not `core_mask`** — masks are correct (the two sub-models need different masks: LLM `popcount=8`→`0xff`, audio `popcount=4`→`0xf`; firmware validates `popcount(mask)==model's embedded core count`). When masks are correct there is no mismatch line; failure is pure `ACK_FAIL`.
- **Not timing / memory-not-released** — inserting 0 s vs 10 s settle delay between loads made no difference (11/11 fail either way).
- **Not application code** — the abort is inside the prebuilt runtime's `rknn3_model_init`, before app inference.

## §5 — Why the host cannot diagnose further
`librknnrt3` strings show the protocol returns only an ACK code; the reason is written to the **EP device-side log**, which the host cannot read:
```
MODEL_SETUP fail, ack = %s
... Please check device side log
... IOCTL_CMD_DUMP fail, ack=%u! Please check server log
... device rejected %s chunk! Check device side log.
```
We tried the host-side EP console `rknn-console rk1820` (PCIe VUART; firmware exposes `PCIE VUART: Initialized (Poll Mode)`): it lists/connects to the device (after freeing the EP from other users), but `rknn-console rk1820 -d 0 shell "<cmd>"` returns no usable output on the V1.0.4 EP firmware. So the EP device-side log is not host-accessible without a physical RISC-V UART.

**Q2: How can we read the EP device-side log over PCIe to obtain the `ACK_FAIL` reason (since the runtime points us to it)? Is there a working `rknn-console` shell / a device-log dump (`IOCTL_CMD_DUMP`)?**

## §6 — Recovery & current mitigation
- **Recovery from regime (B):** a clean host reboot (boot-time `rknn3.service` re-downloads firmware) reliably recovers the EP. **`rknn-smi reset` MUST NOT be used** — it puts the EP into boot state `0xffffffff` ("not in maskrom/loader mode") that survives host reboot and needs a physical 12 V power-cycle. (Reporting this as a separate hazard.)
- **Mitigation for regime (A):** in-process retry (2–3 attempts) around `model_init`/worker start.

**Q3: Is there a firmware fix planned for the model-setup ACK reliability? A V1.0.4 EP firmware that reports 3.03.01 to match the RC runtime?**

## Reproduction
1. RKNN3 V1.0.4 on RK3588 host + RK1828 EP, firmware md5 `37ca5e4e…`.
2. Load a 2-model app (LLM `core_mask=0xff` + audio `core_mask=0xf`) via `rknn3_model_init` repeatedly.
3. Observe intermittent `MODEL_SETUP ACK_FAIL` at the first model init; once it starts failing it tends to fail persistently until a clean reboot.
</content>

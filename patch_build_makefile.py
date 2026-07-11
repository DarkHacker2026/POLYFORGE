#!/usr/bin/env python3
import pathlib

p = pathlib.Path('/home/dark_hacker/hackathon-project/vendor/vortex/build/sim/simx/Makefile')
if p.exists():
    text = p.read_text()
    text = text.replace('dram_sim.cpp', 'dram_sim_stub.cpp')
    text = text.replace('LDFLAGS += -Wl,-rpath,$(THIRD_PARTY_DIR)/ramulator -L$(THIRD_PARTY_DIR)/ramulator -lramulator', '# LDFLAGS')
    p.write_text(text)
    print("Patched build Makefile")
else:
    print("Build Makefile not found")

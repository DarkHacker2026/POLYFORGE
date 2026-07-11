#!/usr/bin/env python3
"""Write the dram_sim stub to the Vortex source tree."""
import pathlib

stub = (
    '#include "dram_sim.h"\n'
    'namespace vortex {\n'
    'class DramSim::Impl {\n'
    'public:\n'
    '  Impl(uint32_t,uint32_t,float){}\n'
    '  void reset(){}\n'
    '  void tick(){}\n'
    '  void send_request(uint64_t,bool,ResponseCallback cb,void* arg){if(cb)cb(arg);}\n'
    '};\n'
    'DramSim::DramSim(uint32_t nc,uint32_t cs,float cr):impl_(new Impl(nc,cs,cr)){}\n'
    'DramSim::~DramSim(){delete impl_;}\n'
    'void DramSim::reset(){impl_->reset();}\n'
    'void DramSim::tick(){impl_->tick();}\n'
    'void DramSim::send_request(uint64_t a,bool w,ResponseCallback cb,void* arg){\n'
    '  impl_->send_request(a,w,cb,arg);\n'
    '}\n'
    '} // namespace vortex\n'
)

dest = pathlib.Path(
    '/home/dark_hacker/hackathon-project/vendor/vortex/sim/common/dram_sim_stub.cpp'
)
dest.write_text(stub)
print(f'Written {dest} ({dest.stat().st_size} bytes)')

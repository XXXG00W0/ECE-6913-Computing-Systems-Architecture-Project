import os
import argparse

MemSize = 1000 # memory size, in reality, the memory size should be 2^32, but for this lab, for the space resaon, we keep it as this large number, but the memory is still 32-bit addressable.

class InsMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        
        with open(ioDir + "\\imem.txt") as im:
            self.IMem = [data.replace("\n", "") for data in im.readlines()]

    def readInstr(self, ReadAddress: int):
        #read instruction memory
        #return 32 bit hex val

        # instruction is big-endian but in reversed order
        # assert isinstance(ReadAddress, int), f"ReadAddress {ReadAddress} is not type of int"
        # assert ReadAddress + 4 <= len(self.IMem), f"ReadAddress {ReadAddress} + 4 = {len(self.IMem)} exceeds instruction memory length {len(self.IMem)}"
        if type(ReadAddress) != int:
            ReadAddress = int(ReadAddress, 2)
        instruction = ""
        for i in range(4):
            instruction =  instruction + self.IMem[ReadAddress+i]
        print(instruction)
        return instruction
          
class DataMem(object):
    def __init__(self, name, ioDir):
        self.id = name
        self.ioDir = ioDir
        with open(ioDir + "\\dmem.txt") as dm:
            self.DMem = [data.replace("\n", "") for data in dm.readlines()]
        # extend data memory size to 1000
        self.DMem.extend(["00000000" for _ in range(MemSize - len(self.DMem))])

    def readInstr(self, ReadAddress):
        #read data memory
        #return 32 bit hex val
        if type(ReadAddress) != int:
            # print(ReadAddress)
            ReadAddress = int(ReadAddress, 2)
        # assert ReadAddress + 4 <= len(self.DMem), f"ReadAddress {ReadAddress} + 4 = {len(self.DMem)} exceeds instruction memory length {len(self.DMem)}"
        memData = ""
        for i in range(4):
            memData = memData + self.DMem[ReadAddress+i]
        print("memory data", memData)
        return memData
        
    def writeDataMem(self, Address, WriteData: str):
        # write data into byte addressable memory
        if type(Address) != int:
            Address = int(Address, 2)
        if type(WriteData) != str:
            WriteData = bin(WriteData)[2:]
        # assert Address + 4 <= len(self.DMem), f"Address {Address} + 4 = {len(self.DMem)} exceeds instruction memory length {len(self.DMem)}"
        for i in range(4):
            self.DMem[Address+i] = WriteData[8*i :8*(i+1)]
                     
    def outputDataMem(self):
        resPath = self.ioDir + "\\" + self.id + "_DMEMResult.txt"
        with open(resPath, "w") as rp:
            rp.writelines([str(data) + "\n" for data in self.DMem])

class RegisterFile(object):
    def __init__(self, ioDir):
        self.outputFile = ioDir + "RFResult.txt"
        self.Registers = ["0"*32 for i in range(32)]
        # self.Registers = [0x0 for i in range(32)]
    
    def readRF(self, Reg_addr: str):
        # Fill in
        reg_addr = int(Reg_addr, 2)
        return self.Registers[reg_addr]
    
    def writeRF(self, Reg_addr, Wrt_reg_data: str):
        # Fill in
        Reg_addr = int(Reg_addr, 2)
        print("write RF: ", Wrt_reg_data, "to ", Reg_addr)
        if Reg_addr != 0:
            self.Registers[Reg_addr] = Wrt_reg_data
        # Reg_addr == 0: write has no effect

         
    def outputRF(self, cycle):
        op = ["-"*70+"\n", "State of RF after executing cycle:" + str(cycle) + "\n"]
        op.extend([str(val)+"\n" for val in self.Registers])
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.outputFile, perm) as file:
            file.writelines(op)

class State(object):
    def __init__(self):
        self.IF = {"nop": False, "PC": 0}
        self.ID = {"nop": False, "Instr": 0}
        self.EX = {"nop": False, "Read_data1": 0, "Read_data2": 0, "Imm": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "is_I_type": False, "rd_mem": 0, 
                   "wrt_mem": 0, "alu_op": 0, "wrt_enable": 0}
        self.MEM = {"nop": False, "ALUresult": 0, "Store_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "rd_mem": 0, 
                   "wrt_mem": 0, "wrt_enable": 0}
        self.WB = {"nop": False, "Wrt_data": 0, "Rs": 0, "Rt": 0, "Wrt_reg_addr": 0, "wrt_enable": 0}

class Core(object):
    def __init__(self, ioDir, imem: InsMem, dmem: DataMem):
        self.myRF = RegisterFile(ioDir)
        self.cycle = 0
        self.halted = False
        self.ioDir = ioDir
        self.state = State()
        self.nextState = State()
        self.ext_imem = imem
        self.ext_dmem = dmem

        # Performance metrics purpose
        self.instruction_count = 0

class SingleStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(SingleStageCore, self).__init__(ioDir + "\\SS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_SS.txt"

    def getControlOutput(self, opcode):
        
        controlOutput = {"ALUSrc": "X", "MemtoReg": "X", "RegWrite": "X", "MemRead": "X", 
                            "MemWrite": "X", "Branch": "X", "ALUOp1": "X", "ALUOp0":"X"}
        
        # R-format: add sub xor or and
        if opcode == "0110011":
            controlOutput = {"ALUSrc": "0", "MemtoReg": "0", "RegWrite": "1", "MemRead": "0", 
                             "MemWrite": "0", "Branch": "0", "ALUOp1": "1", "ALUOp0": "0"}
        # I-type: addi xori ori andi
        elif opcode == "0010011":
            controlOutput = {"ALUSrc": "1", "MemtoReg": "0", "RegWrite": "1", "MemRead": "0", 
                             "MemWrite": "0", "Branch": "0", "ALUOp1": "1", "ALUOp0": "0"}
            self.state.EX["is_I_type"] = 1
        # load: lw
        elif opcode == "0000011":
            controlOutput = {"ALUSrc": "1", "MemtoReg": "1", "RegWrite": "1", "MemRead": "1", 
                             "MemWrite": "0", "Branch": "0", "ALUOp1": "0", "ALUOp0": "0"}
        # store: sw
        elif opcode == "0100011":
            controlOutput = {"ALUSrc": "1", "MemtoReg": "X", "RegWrite": "0", "MemRead": "0", 
                             "MemWrite": "1", "Branch": "0", "ALUOp1": "0", "ALUOp0": "0"}
        # b type branch: beq bne
        elif opcode == "1100011":
            controlOutput = {"ALUSrc": "0", "MemtoReg": "X", "RegWrite": "0", "MemRead": "0", 
                             "MemWrite": "0", "Branch": "1", "ALUOp1": "0", "ALUOp0": "1"}
        # jtype
        elif opcode == "1101111":
            controlOutput = {"ALUSrc": "X", "MemtoReg": "0", "RegWrite": "1", "MemRead": "0", 
                             "MemWrite": "0", "Branch": "0", "ALUOp1": "X", "ALUOp0": "X"}
        # halt
        elif all(c == "1" for c in opcode):
            controlOutput = {"ALUSrc": "X", "MemtoReg": "X", "RegWrite": "X", "MemRead": "X", 
                             "MemWrite": "X", "Branch": "X", "ALUOp1": "X", "ALUOp0": "X"}
            # self.halted = True
            # self.nextState.IF["nop"] = True
        else:
            print(f"Opcode {opcode} does not belongs to any valid instruction")
        
        return controlOutput
    
    def immDecode(self, opcode, instructionLH):
        # default value
        imm = "0" * 32
        # i-type: ADDI XORI ORI ANDI or LW
        # print("Instruction Low -> High bit: ", instructionLH)
        if opcode == "0010011" or opcode == "0000011":
            imm = instructionLH[20:]
        # b-type: BEQ BNE
        elif opcode == "1100011":
            #     imm[4:1]              imm[10:5]            imm[11]              imm[12]
            imm = instructionLH[8:12] + instructionLH[25:31] + instructionLH[7] + instructionLH[31]
        # s-type: SW
        elif opcode == "0100011":
            #     imm[4:0]              imm[11:5]   
            imm = instructionLH[7:12] + instructionLH[25:]
        # j-type: JAL
        elif opcode == "1101111":
            # imm[20|10:1|11|19:12] 
            #     imm[10:1]              imm[11]             imm[19:12]             imm[20]                                      
            imm = instructionLH[21:31] + instructionLH[20] + instructionLH[12:20] + instructionLH[31]
        imm = imm[::-1]
        print("decode imm:", imm)
        return imm
        
    def getALUControl(self, opcode, funct3, funct7, aluOp):
        # find ALU control signal based on ALUOp, opcode, funct3 and funct7

        match opcode:
            case "1100011": # B-type BEQ BNE
                return "0110" # SUB
            case "0100011": # S-type SW
                return "0010" # ADD
            case "0110011": # R-type
                match funct3:
                    case "000": # ADD | SUB
                        match funct7:
                            case "0100000": return "0110" # SUB
                            case "0000000": return "0010" # ADD
                            case _: return "1111" # default
                    case "100": return "0111" # XOR
                    case "110": return "0001" # OR
                    case "111": return "0000" # AND
                    case _: return "1111" # default
            case "0010011": # I-type w/o LW
                match funct3:
                    case "000": return "0010" # ADD
                    case "100": return "0111" # XOR
                    case "110": return "0001" # OR
                    case "111": return "0000" # AND
                    case _: return "1111" # default
            case "0000011": # LW
                match funct3:
                    case "000": return "0010" # ADD
                    case _: return "1111" # default
            case _: return "1111" # default

    def getALUout(self, aluControl, readData1, readData2):
        result = 0
        if type(readData1) != int: readData1 = self.to_decimal(readData1)
        if type(readData2) != int: readData2 = self.to_decimal(readData2)
        # add
        if aluControl == "0010": result = readData1 + readData2
        # sub
        elif aluControl == "0110": result = readData1 - readData2
        # AND
        elif aluControl == "0000": result = readData1 & readData2
        # OR
        elif aluControl == "0001": result = readData1 | readData2
        # XOR to-do 
        elif aluControl == "0111": result = readData1 ^ readData2
        else: print(f"Unknown alu control {aluControl}")
        # return string reprentation of the calculation result (without the "0b" prefix)
        
        # rs1 == rs2 Zero = 1 | rs1 != rs2 Zero = 0
        Zero = "1" if result == 0 else "0"
        return self.to_binary(result), Zero
    
    def to_decimal(self, binary_string):
        if "X" in binary_string:
            return "X"
        # check the sign 
        is_negative = binary_string[0] == '1'
        
        if is_negative:
            # convert to positive binary repr
            inverted = ''.join('1' if bit == '0' else '0' for bit in binary_string)  # flip bits
            as_positive_binary = bin(int(inverted, 2) + 1)[2:]  # plus 1
            decimal = -int(as_positive_binary, 2)  # to decimal and add negative sign
        else:
            decimal = int(binary_string, 2)  # positive binary directly to decimal
        
        return decimal
    
    def to_binary(self, number, bits=32):
        return bin(number & (2**bits - 1))[2:].zfill(bits)
    
    def instructionDebug(self, instructionLH):
        
        opcode = instructionLH[0:7][::-1]
        rd =  instructionLH[7:12][::-1]
        funct3 =  instructionLH[12:15][::-1]
        rs1 =  instructionLH[15:20][::-1]
        rs2 = instructionLH[20:25][::-1]
        funct7 = instructionLH[25:][::-1]

        imm = self.immDecode(opcode, instructionLH)
        if opcode == "0110011":
            match funct7:
                case "0100000": print(f"rd:{rd} = rs1:{rs1} SUB rs2:{rs2}: ")
                case "0000000":
                    match funct3:
                        case "000": print(f"rd:{rd} = rs1:{rs1} ADD rs2:{rs2}: ")
                        case "100": print(f"rd:{rd} = rs1:{rs1} XOR rs2:{rs2}: ")
                        case "110": print(f"rd:{rd} = rs1:{rs1} OR rs2:{rs2}: ")
                        case "111": print(f"rd:{rd} = rs1:{rs1} AND rs2:{rs2}: ")
                        case _: print(f"Invalid funct3 {funct3}")
                case _: print(f"Invalid funct7 {funct7}")
            print(f"R type, opcode: {opcode}, rd: {rd}, funct3: {funct3}, rs1: {rs1}, rs2: {rs2}, funct7: {funct7}") 
        elif opcode in ["0000011", "0010011"]:
            match opcode:
                case "0010011":
                    match funct3:
                        case "000": print(f"rd:{rd} = rs1:{rs1} ADDI imm:{imm}")
                        case "100": print(f"rd:{rd} = rs1:{rs1} XORI imm:{imm}")
                        case "110": print(f"rd:{rd} = rs1:{rs1} ORI imm:{imm}")
                        case "111": print(f"rd:{rd} = rs1:{rs1} ANDI imm:{imm}")
                        case _: print(f"Invalid funct3 {funct3}")
                case "0000011": print(f"rd:{rd} <= LW (imm:{imm})rs1:{rs1}: ")
            print(f"I type, opcode: {opcode}, rd: {rd}, funct3: {funct3}, rs1: {rs1}, imm: {imm}")
        elif opcode == "1101111":
            print(f"JAL jump to (imm:{imm})rd{rd}")
            print(f"J type, opcode: {opcode}, rd: {rd}, imm[20|10:1|11|19:12]: {instructionLH[12:][::-1]}->{imm}")
        elif opcode == "1100011":
            match funct3:
                case "000": print(f"BEQ if rs1:{rs1} == rs2:{rs2} goto imm:{imm}: ")
                case "001": print(f"BNE if rs1:{rs1} != rs2:{rs2} goto imm:{imm}: ")
                case _: print(f"Invalid funct3 {funct3}")
            print(f"B type, opcode: {opcode}, imm[4:1|11]: {rd}->{imm}, {funct3}, rs1: {rs1}, rs2: {rs2}, imm[12|10:5]: {funct7}")
        elif opcode == "0100011":
            print(f"SW DMEM[(imm:{imm})rs1:{rs1}] <- rs2:{rs2}")
            print(f"S type, opcode: {opcode}, imm[4:0]: {rd}->{imm}, {funct3}, rs1: {rs1}, rs2: {rs2}, imm[11:5]: {funct7}")
        elif opcode == "1111111":
            print("HALT")
        else:
            print("invalid opcode ", opcode, end=" ")

    def step(self):
        # Your implementation
        
        # IF stage
        PC: int = self.state.IF["PC"]
        print("CYCLE ", self.cycle, "PC ", PC)
        # High bit -> Low bit instruction for viewing and debugging
        instructionHL: str = self.ext_imem.readInstr(PC)
        # Low bit -> high bit instruction for decoding
        instructionLH = instructionHL[::-1]
        print("Instruction High -> Low bit: ", instructionHL)
        print("Instruction Low -> High bit: ", instructionLH)
        
        
        # increment instruction count by 1 if not in HALT state
        if self.state.IF["nop"] == False:
            self.instruction_count += 1

        # ID stage
        self.state.ID["Instr"] == instructionLH
        control: str = instructionLH[0:7][::-1] # opcode
        writeRegister: str = instructionLH[7:12][::-1] # rd
        aluControl: str = instructionLH[30] + instructionLH[12:16][::-1] # funct3 + instruction[30]
        readRegister1: str = instructionLH[15:20][::-1] # rs1
        readRegister2: str = instructionLH[20:25][::-1] # rs2

        # instruction decode
        opcode = control
        rd =  instructionLH[7:12][::-1]
        funct3 =  instructionLH[12:15][::-1]
        rs1 =  instructionLH[15:20][::-1]
        rs2 = instructionLH[20:25][::-1]
        funct7 = instructionLH[25:][::-1]
        self.instructionDebug(instructionLH)

        # HALT
        if opcode == "1111111":
            self.nextState.IF["nop"] = True
            self.nextState.IF["PC"] = PC
        # not HALT
        else:

            # imm decode
            # instruction[0 -> 31]]
            imm = self.immDecode(opcode, instructionLH)

            # EX stage
            print(control)
            controlOutput = self.getControlOutput(opcode) # control signal
            
            # update state
            self.state.EX["Read_data1"] = self.myRF.readRF(readRegister1)
            self.state.EX["Read_data2"] = self.myRF.readRF(readRegister2)
            self.state.EX["imm"] = imm
            self.state.EX["Rs"] = readRegister1
            self.state.EX["Rt"] = readRegister2
            self.state.EX["Wrt_reg_addr"] = writeRegister
            self.state.EX["rd_mem"] = controlOutput["MemRead"]
            self.state.EX["wrt_mem"] = controlOutput["MemWrite"]
            self.state.EX["alu_op"] = str(controlOutput["ALUOp1"]) + str(controlOutput["ALUOp0"])
            self.state.EX["wrt_enable"] = controlOutput["RegWrite"]

            # ? Not used in single cycle
            if self.state.EX["is_I_type"] == "1":
                pass

            # pre-ALU Mux
            if controlOutput["ALUSrc"] == "0":
                readData2 = self.state.EX["Read_data2"]
            elif controlOutput["ALUSrc"] == "1":
                readData2 = imm
            else:
                readData2 = "X"
                print(f"Unknown ALUSrc signal: {controlOutput['ALUSrc']}")

            # ALU control
            aluControl = self.getALUControl(opcode, funct3, funct7, self.state.EX["alu_op"])

            # ALU 
            readData1 = self.state.EX["Read_data1"]
            aluResult, Zero = self.getALUout(aluControl, readData1, readData2)

            # MUX 0: next PC, 1: adderSum, Control: Branch & ALU_Zero

            # check command type: BEQ BNE or JAL 
            match opcode:
                case "1100011": # B-type
                    match funct3:
                        case "000": pass # BEQ No need to change Zero's value
                        case "001": Zero = "1" if Zero == "0" else "0" # BNE flip Zero's bit
                case "1101111": # J-type
                    Zero = "1" # JAL Zero is always 1
                    # self.myRF.writeRF(rd, PC + 4) # write the pc position of the next command to rd

            if "X" in controlOutput["Branch"]:
                muxControl = "X"
            else:
                muxControl = bin(int(controlOutput["Branch"], 2) & int(Zero, 2))[2:]
            # IF stage: PC increment or jump or branch
            if muxControl == "1":
                # PC + imm<<1
                self.nextState.IF["PC"] = PC + (self.to_decimal(imm) << 1)
            elif muxControl == "0":
                # PC + 4 No B-type or J-type
                self.nextState.IF["PC"] = PC + 4
            else:
                print(f"Invalid mux control signal: {muxControl}")

            # MEM stage
            self.state.MEM["ALUresult"] = aluResult
            self.state.MEM["Store_data"] = self.state.EX["Read_data2"] # 不清楚
            self.state.MEM["Rs"] = self.state.EX["Rs"]
            self.state.MEM["Rt"] = self.state.EX["Rt"]
            self.state.MEM["Wrt_reg_addr"] = self.state.EX["Wrt_reg_addr"]
            self.state.MEM["rd_mem"] = self.state.EX["rd_mem"]
            self.state.MEM["wrt_mem"] = self.state.EX["wrt_mem"]
            self.state.MEM["wrt_enable"] = self.state.EX["wrt_enable"]

            # read memory
            if self.state.MEM["rd_mem"] == "1":
                memReadDataHL = self.ext_dmem.readInstr(self.state.MEM["ALUresult"])
                memReadDataLH = memReadDataHL[::-1]
            # write memory
            if self.state.MEM["wrt_mem"] == "1":
                writeMemDataHL = self.state.MEM["Store_data"]
                self.ext_dmem.writeDataMem(self.state.MEM["ALUresult"], writeMemDataHL)

            # post data memory Mux
            if controlOutput["MemtoReg"] == "1":
                writeRegisterData = memReadDataHL
            elif controlOutput["MemtoReg"] == "0" and controlOutput["ALUSrc"] != "X":
                writeRegisterData = aluResult
            # JAL write PC + 4 to rd (writeRegister)
            elif controlOutput["MemtoReg"] == "0" and controlOutput["ALUSrc"] == "X":
                writeRegisterData = PC + 4

            # Write data to Registers
            if controlOutput["RegWrite"] == "1":
                # print(controlOutput["RegWrite"], readRegister1, writeData)
                self.myRF.writeRF(writeRegister, writeRegisterData)
            
        # self.halted = True
        # print(self.state.IF["nop"])
        if self.state.IF["nop"]:
            self.halted = True
            
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
        print(self.myRF.Registers)
        print(self.ext_dmem.DMem[:16])
            
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1
        self.nextState = State()


    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.append("IF.PC: " + str(state.IF["PC"]) + "\n")
        printstate.append("IF.nop: " + str(state.IF["nop"]) + "\n")
        
        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)

class FiveStageCore(Core):
    def __init__(self, ioDir, imem, dmem):
        super(FiveStageCore, self).__init__(ioDir + "\\FS_", imem, dmem)
        self.opFilePath = ioDir + "\\StateResult_FS.txt"

    def step(self):
        # Your implementation
        # --------------------- WB stage ---------------------
        
        
        
        # --------------------- MEM stage --------------------
        
        
        
        # --------------------- EX stage ---------------------
        
        
        
        # --------------------- ID stage ---------------------
        
        
        
        # --------------------- IF stage ---------------------
        
        self.halted = True
        if self.state.IF["nop"] and self.state.ID["nop"] and self.state.EX["nop"] and self.state.MEM["nop"] and self.state.WB["nop"]:
            self.halted = True
        
        self.myRF.outputRF(self.cycle) # dump RF
        self.printState(self.nextState, self.cycle) # print states after executing cycle 0, cycle 1, cycle 2 ... 
        
        self.state = self.nextState #The end of the cycle and updates the current state with the values calculated in this cycle
        self.cycle += 1

    def printState(self, state, cycle):
        printstate = ["-"*70+"\n", "State after executing cycle: " + str(cycle) + "\n"]
        printstate.extend(["IF." + key + ": " + str(val) + "\n" for key, val in state.IF.items()])
        printstate.extend(["ID." + key + ": " + str(val) + "\n" for key, val in state.ID.items()])
        printstate.extend(["EX." + key + ": " + str(val) + "\n" for key, val in state.EX.items()])
        printstate.extend(["MEM." + key + ": " + str(val) + "\n" for key, val in state.MEM.items()])
        printstate.extend(["WB." + key + ": " + str(val) + "\n" for key, val in state.WB.items()])

        if(cycle == 0): perm = "w"
        else: perm = "a"
        with open(self.opFilePath, perm) as wf:
            wf.writelines(printstate)


if __name__ == "__main__":
     
    #parse arguments for input file location
    parser = argparse.ArgumentParser(description='RV32I processor')
    parser.add_argument('--iodir', default="", type=str, help='Directory containing the input files.')
    args = parser.parse_args()

    ioDir = os.path.abspath(args.iodir)
    print("IO Directory:", ioDir)

    imem = InsMem("Imem", ioDir)
    dmem_ss = DataMem("SS", ioDir)
    dmem_fs = DataMem("FS", ioDir)
    
    ssCore = SingleStageCore(ioDir, imem, dmem_ss)
    fsCore = FiveStageCore(ioDir, imem, dmem_fs)

    while(True):
        if not ssCore.halted:
            ssCore.step()
        
        if not fsCore.halted:
            fsCore.step()

        if ssCore.halted and fsCore.halted:
            break
    
    # dump SS and FS data mem.
    dmem_ss.outputDataMem()
    dmem_fs.outputDataMem()

    # write performance metrics
    ssCoreCPI = ssCore.cycle / ssCore.instruction_count
    fsCoreCPI = fsCore.cycle / (fsCore.instruction_count + 1e-40) # Placeholder to prevent ZeroDivisionError
    with open("PerformanceMetrics.txt", 'w') as pmf:
        pmf.write("Performance of Single Stage:\n")
        pmf.write(f"#Cycles -> {ssCore.cycle}\n")
        pmf.write(f"#Instructions -> {ssCore.instruction_count}\n")
        pmf.write(f"CPI -> {ssCoreCPI}\n")
        pmf.write(f"IPC -> {1 / ssCoreCPI}\n\n")

        pmf.write("Performance of Five Stage:\n")
        pmf.write(f"#Cycles -> {fsCore.cycle}\n")
        pmf.write(f"#Instructions -> {fsCore.instruction_count}\n")
        pmf.write(f"CPI -> {fsCoreCPI}\n")
        pmf.write(f"IPC -> {1 / fsCoreCPI}\n")
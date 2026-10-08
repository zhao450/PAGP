from tree_calculate import GP_evolve_S_PC
from dataclasses import dataclass
from typing import Any, List, Tuple

@dataclass
class Cell:
    id: int
    id_fa_ma_op: int
    priority_data: Any 
    priority_rank: int 

def computer_PC(population,best_index,rou_node_data,seq_node_data):
    pop_pc=[]
    best_ind = population[best_index]
    rou_best: List[Tuple[Cell, ...]] = [None] * 20
    seq_best: List[Tuple[Cell, ...]] = [None] * 20
    rou=[[0 for _ in range(20)] for _ in range(len(population))]
    seq=[[0 for _ in range(20)] for _ in range(len(population))]
    for i in range(20):
        rou_row_best = [Cell(id=k, id_fa_ma_op=0,priority_data=None,priority_rank=99) for k in range(3)]
        seq_row_best = [Cell(id=k, id_fa_ma_op=0,priority_data=None,priority_rank=99) for k in range(3)]
        for j in range(3):

            machine=int(rou_node_data[i][j]['machine'][1:])
            niq=int(rou_node_data[i][j]['NIQ'])
            mwt=int(rou_node_data[i][j]['MWT'])
            pt=int(rou_node_data[i][j]['PT'])
            npt=int(rou_node_data[i][j]['NPT'])
            wiq=int(rou_node_data[i][j]['WIQ'])
            owt=int(rou_node_data[i][j]['OWT'])
            tis=int(rou_node_data[i][j]['TIS'])
            wkr=int(rou_node_data[i][j]['WKR'])
            nor=int(rou_node_data[i][j]['NOR'])
            slack=int(rou_node_data[i][j]['SLACK'])
            machine_data=[niq,mwt,pt,npt,wiq,owt,tis,wkr,nor,slack]
            best_ind_rou=GP_evolve_S_PC(machine_data, best_ind[0])
            rou_row_best[j].id_fa_ma_op=machine
            rou_row_best[j].priority_data=best_ind_rou

            op=int(seq_node_data[i][j]['job_id'])
            niq=int(seq_node_data[i][j]['NIQ'])
            mwt=int(seq_node_data[i][j]['MWT'])
            pt=int(seq_node_data[i][j]['PT'])
            npt=int(seq_node_data[i][j]['NPT'])
            wiq=int(seq_node_data[i][j]['WIQ'])
            owt=int(seq_node_data[i][j]['OWT'])
            tis=int(seq_node_data[i][j]['TIS'])
            wkr=int(seq_node_data[i][j]['WKR'])
            nor=int(seq_node_data[i][j]['NOR'])
            slack=int(seq_node_data[i][j]['SLACK'])
            op_data=[niq,mwt,pt,npt,wiq,owt,tis,wkr,nor,slack]
            best_ind_seq=GP_evolve_S_PC(op_data, best_ind[1])
            seq_row_best[j].id_fa_ma_op=op
            seq_row_best[j].priority_data=best_ind_seq

        order_rou = sorted(range(len(rou_row_best)),key=lambda idx: (rou_row_best[idx].priority_data, rou_row_best[idx].id))
        order_seq = sorted(range(len(seq_row_best)),key=lambda idx: (seq_row_best[idx].priority_data, seq_row_best[idx].id))
        for rank, idx in enumerate(order_rou, start=1):
            rou_row_best[idx].priority_rank = rank
        for rank, idx in enumerate(order_seq, start=1):
            seq_row_best[idx].priority_rank = rank

        rou_best[i]=tuple(rou_row_best)
        seq_best[i]=tuple(seq_row_best)

    for n in range(len(population)):
        for k in range(20):
            rou_row = [Cell(id=z, id_fa_ma_op=0,priority_data=None,priority_rank=99) for z in range(3)]
            seq_row = [Cell(id=z, id_fa_ma_op=0,priority_data=None,priority_rank=99) for z in range(3)]
            for m in range(3):

                machine=int(rou_node_data[k][m]['machine'][1:])
                niq=int(rou_node_data[k][m]['NIQ'])
                mwt=int(rou_node_data[k][m]['MWT'])
                pt=int(rou_node_data[k][m]['PT'])
                npt=int(rou_node_data[k][m]['NPT'])
                wiq=int(rou_node_data[k][m]['WIQ'])
                owt=int(rou_node_data[k][m]['OWT'])
                tis=int(rou_node_data[k][m]['TIS'])
                wkr=int(rou_node_data[k][m]['WKR'])
                nor=int(rou_node_data[k][m]['NOR'])
                slack=int(rou_node_data[k][m]['SLACK'])
                machine_data=[niq,mwt,pt,npt,wiq,owt,tis,wkr,nor,slack]
                best_ind_rou=GP_evolve_S_PC(machine_data, population[n][0])
                rou_row[m].id_fa_ma_op=machine
                rou_row[m].priority_data=best_ind_rou

                op=int(seq_node_data[k][m]['job_id'])
                niq=int(seq_node_data[k][m]['NIQ'])
                mwt=int(seq_node_data[k][m]['MWT'])
                pt=int(seq_node_data[k][m]['PT'])
                npt=int(seq_node_data[k][m]['NPT'])
                wiq=int(seq_node_data[k][m]['WIQ'])
                owt=int(seq_node_data[k][m]['OWT'])
                tis=int(seq_node_data[k][m]['TIS'])
                wkr=int(seq_node_data[k][m]['WKR'])
                nor=int(seq_node_data[k][m]['NOR'])
                slack=int(seq_node_data[k][m]['SLACK'])
                op_data=[niq,mwt,pt,npt,wiq,owt,tis,wkr,nor,slack]
                best_ind_seq=GP_evolve_S_PC(op_data,population[n][1])
                seq_row[m].id_fa_ma_op=op
                seq_row[m].priority_data=best_ind_seq
            order_rou = sorted(range(len(rou_row)),key=lambda idx: (rou_row[idx].priority_data, rou_row[idx].id))
            order_seq = sorted(range(len(seq_row)),key=lambda idx: (seq_row[idx].priority_data, seq_row[idx].id))
            for rank, idx in enumerate(order_rou, start=1):
                rou_row[idx].priority_rank = rank
            for rank, idx in enumerate(order_seq, start=1):
                seq_row[idx].priority_rank = rank

            for candidate in range(3):
                if rou_row[candidate].priority_rank==1:
                    pop_rou_name=rou_row[candidate].id_fa_ma_op
                    break
            for candidate in range(3):
                if seq_row[candidate].priority_rank==1:
                    pop_seq_name=seq_row[candidate].id_fa_ma_op
                    break

            for candidate in range(3):
                if rou_best[k][candidate].id_fa_ma_op==pop_rou_name:
                    ref=rou_best[k][candidate].priority_rank
                    rou[n][k]=ref
                    break
            for candidate in range(3):
                if seq_best[k][candidate].id_fa_ma_op==pop_seq_name:
                    ref=seq_best[k][candidate].priority_rank
                    seq[n][k]=ref
                    break

        pop_pc.append(rou[n]+seq[n])
    return pop_pc

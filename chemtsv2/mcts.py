import os
import sys
import time

import numpy as np
import pandas as pd
import pickle

from chemtsv2.utils import (
    generate_smiles_as_token_index,
    build_smiles_from_token_index,
    evaluate_node,
    get_token_to_add,
    get_expanded_node_index,
    back_propagation,
)

from chemtsv2.SmilesEnumerator import SmilesEnumerator
from chemtsv2.Synonymous_processing  import  smi_tokenizer,standardize_smiles
import csv

class State:
    def __init__(self, position=["&"], parent=None):
        self.position = position
        self.visits = 0
        self.total_reward = 0
        self.parent_node = parent
        self.child_nodes = []

    def clone(
        self,
        include_visit=False,
        include_total_reward=False,
        include_parent_node=False,
        include_child_node=False,
    ):
        st = State()
        st.position = self.position[:]
        st.visits = self.visits if include_visit else 0
        st.total_reward = self.total_reward if include_total_reward else 0
        st.parent_node = self.parent_node if include_parent_node else None
        st.child_nodes = self.child_nodes if include_child_node else []
        return st

    def add_position(self, m):
        self.position.append(m)


class Node:
    def __init__(self, position=None, state=None, conf=None):
        self.position = position
        self.state = state
        self.conf = conf

    def select_node(self, policy_evaluator, logger):
        score_list = []
        logger.debug("UCB:")
        for i in range(len(self.state.child_nodes)):
            score = policy_evaluator.evaluate(self.state.child_nodes[i].state, self.conf)
            score_list.append(score)
            logger.debug(f"{self.state.child_nodes[i].position} {score}")
        m = np.amax(score_list)
        indices = np.nonzero(score_list == m)[0]
        ind = int(self.conf["random_generator"].choice(indices))
        s = self.state.child_nodes[ind]
        logger.debug(f"\nindex {ind} {self.position} {m}")
        return s

    def add_node(self, m, state):
        state.parent_node = self
        node = Node(position=m, state=state, conf=self.conf)
        self.state.child_nodes.append(node)

    def update(self, reward):
        self.state.visits += 1
        self.state.total_reward += reward




class MCTS:
    def __init__(
        self,
        root_state,
        conf,
        tokens,
        model,
        reward_calculator,
        policy_evaluator,
        logger,
    ):
        self.start_time = time.time()
        self.rootnode = Node(state=root_state, conf=conf)
        self.conf = conf
        self.tokens = tokens
        self.model = model
        self.reward_calculator = reward_calculator
        self.policy_evaluator = policy_evaluator
        self.logger = logger

        self.valid_smiles_list = []
        self.depth_list = []
        self.objective_values_list = []
        self.reward_values_list = []
        self.elapsed_time_list = []
        self.generated_dict = {}  # dictionary of generated compounds
        self.generated_id_list = []
        self.filter_check_list = []
        self.heavy_atom_count_list=[]
        self.total_valid_num = 0






        if conf["batch_reward_calculation"]:
            self.obj_column_names = [
                f.__name__ for f in self.reward_calculator.get_batch_objective_functions()
            ]
        else:
            self.obj_column_names = [
                f.__name__ for f in self.reward_calculator.get_objective_functions(self.conf)
            ]
        self.output_path = os.path.join(conf["output_dir"], f"result_C{conf['c_val']}.csv")
        if os.path.exists(self.output_path) and not conf["restart"]:
            sys.exit(
                f"[ERROR] {self.output_path} already exists. Please specify a different file name."
            )

        self.gid = 0
        self.loop_counter_for_selection = 0
        self.loop_counter_for_expansion = 0
        self.expanded_before = {}

        if conf["threshold_type"] == "time":
            self.threshold = time.time() + 3600 * conf["hours"]
        elif conf["threshold_type"] == "generation_num":
            self.threshold = conf["generation_num"]
        else:
            sys.exit("[ERROR] Specify 'threshold_type': [time, generation_num]")

        self.synonymous_threshold_num = conf["synonymous_threshold_num"]


        self.synonymous_counter=0

    def flush(self):
        df = pd.DataFrame({
            "generated_id": self.generated_id_list,
            "smiles": self.valid_smiles_list,
            "reward": self.reward_values_list,
            "depth": self.depth_list,
            "elapsed_time": self.elapsed_time_list,
            "is_through_filter": self.filter_check_list,
        })
        # 强制 reward 列为数值类型
        df["reward"] = pd.to_numeric(df["reward"], errors="coerce")
        df_obj = pd.DataFrame(self.objective_values_list, columns=self.obj_column_names)
        df = pd.concat([df, df_obj], axis=1)
        if os.path.exists(self.output_path):
            df.to_csv(self.output_path, mode="a", index=False, header=False)
        else:
            df.to_csv(self.output_path, mode="w", index=False)
        self.logger.info(f"save results at {self.output_path}")


        self.synonymous_counter += len(self.valid_smiles_list)

        self.generated_id_list.clear()
        self.valid_smiles_list.clear()
        self.reward_values_list.clear()
        self.depth_list.clear()
        self.elapsed_time_list.clear()
        self.filter_check_list.clear()
        self.objective_values_list.clear()








    def search(self):
        """initialization of search tree"""
        ckpt_path = os.path.join(self.conf["output_dir"], self.conf["checkpoint_file"])
        if self.conf["restart"] and os.path.exists(ckpt_path):
            self.logger.info(f"Load the checkpoint file from {ckpt_path}")
            self.load_checkpoint()

        while (
            time.time() if self.conf["threshold_type"] == "time" else self.total_valid_num
        ) <= self.threshold:
            node = self.rootnode
            state = node.state.clone()

            """selection step"""
            node_pool = []
            while node.state.child_nodes != []:
                node = node.select_node(self.policy_evaluator, self.logger)
                state.add_position(node.position)
            self.logger.info(f"state position: {state.position}")

            self.logger.debug(
                f"infinite loop counter (selection): {self.loop_counter_for_selection}"
            )
            if node.position == "\n":
                back_propagation(node, reward=-1.0)
                self.loop_counter_for_selection += 1
                if (
                    self.loop_counter_for_selection
                    > self.conf["infinite_loop_threshold_for_selection"]
                ):
                    self.flush()
                    sys.exit(
                        "[WARN] Infinite loop is detected in the selection step. Change hyperparameters or RNN model."
                    )
                continue
            else:
                self.loop_counter_for_selection = 0

            """expansion step"""
            expanded = get_expanded_node_index(
                self.model,
                state.position,
                self.tokens,
                self.logger,
                threshold=self.conf["expansion_threshold"],
            )
            self.logger.debug(
                f"infinite loop counter (expansion): {self.loop_counter_for_expansion}"
            )
            if set(expanded) == self.expanded_before:
                self.loop_counter_for_expansion += 1
                if (
                    self.loop_counter_for_expansion
                    > self.conf["infinite_loop_threshold_for_expansion"]
                ):
                    self.flush()
                    sys.exit(
                        "[WARN] Infinite loop is detected in the expansion step. Change hyperparameters or RNN model."
                    )
            else:
                self.loop_counter_for_expansion = 0
            self.expanded_before = set(expanded)

            """simulation step"""
            new_compound = []
            nodeadded = []
            for _ in range(self.conf["simulation_num"]):
                nodeadded_tmp = get_token_to_add(expanded, self.tokens, self.logger)
                nodeadded.extend(nodeadded_tmp)
                for n in nodeadded_tmp:
                    position_tmp = state.position + [n]
                    generated_token_indexes = generate_smiles_as_token_index(
                        self.model, position_tmp, self.tokens, self.conf
                    )
                    new_compound.append(
                        build_smiles_from_token_index(
                            generated_token_indexes,
                            self.tokens,
                            use_selfies=self.conf["use_selfies"],
                        )
                    )

            _gids = list(range(self.gid, self.gid + len(new_compound)))
            self.gid += len(new_compound)

            self.logger.debug(f"Added nodes: {nodeadded}")
            self.logger.debug(f"Generated molecules dictionary: {self.generated_dict}")
            self.logger.info(f"New compounds: {new_compound}")
            self.logger.debug(
                "\n"
                + "\n".join([
                    f"Index {i}: {c} (last character {c[-1]}, line break: "
                    + str(c[-1] == "\n")
                    + ")"
                    for i, c in enumerate(new_compound)
                ])
            )
            (
                node_index,
                objective_values,
                valid_smiles,
                generated_id_list,
                filter_check_list,
            ) = evaluate_node(
                new_compound,
                self.generated_dict,
                self.reward_calculator,
                self.conf,
                self.logger,
                _gids,
            )

            if len(valid_smiles) == 0:
                back_propagation(node, reward=-1.0)
                continue

            valid_num = len(valid_smiles)
            self.total_valid_num += valid_num
            self.valid_smiles_list.extend(valid_smiles)
            depth = len(state.position)
            self.depth_list.extend([depth for _ in range(valid_num)])
            elapsed_time = round(time.time() - self.start_time, 1)
            self.elapsed_time_list.extend([elapsed_time for _ in range(valid_num)])
            self.objective_values_list.extend(objective_values)
            self.generated_id_list.extend(generated_id_list)
            self.filter_check_list.extend(filter_check_list)




            self.logger.info(f"Number of valid SMILES in this cycle: {self.total_valid_num}")
            self.logger.debug(
                f"\nNode index: {node_index} \nObjective_values: {objective_values} \nValid smiles: {valid_smiles} \nTime {elapsed_time}"
            )

            re_list = []
            atom_checked = []
            for i in range(len(node_index)):
                m = node_index[i]
                atom = nodeadded[m]
                state_clone = state.clone(include_visit=True, include_total_reward=True)

                if atom not in atom_checked:
                    node.add_node(atom, state_clone)
                    node_pool.append(node.state.child_nodes[len(atom_checked)])
                    atom_checked.append(atom)
                else:
                    node_pool.append(node.state.child_nodes[atom_checked.index(atom)])

                self.logger.debug(
                    "\n"
                    + "\n".join([
                        f"Child node position: {c.position}" for c in node.state.child_nodes
                    ])
                )

                re = (
                    -1
                    if atom == "\n"
                    else self.reward_calculator.calc_reward_from_objective_values(
                        values=objective_values[i], conf=self.conf,smiles=valid_smiles[i]
                    )
                )
                if self.conf["include_filter_result_in_reward"]:
                    re *= filter_check_list[i]

                re_list.append(re)
                self.logger.debug(f"\nAdded atom: {atom} \nReward list: {re_list}")
            self.reward_values_list.extend(re_list)

            """backpropation step"""
            for i in range(len(node_pool)):
                node = node_pool[i]
                back_propagation(node, reward=re_list[i])

            self.logger.debug(
                "\n"
                + "\n".join([
                    f"Child node position: {c.position} Total reward: {c.state.total_reward} Visit count: {c.state.visits}"
                    for c in node_pool
                ])
            )

            if (
                len(self.valid_smiles_list) > self.conf["flush_threshold"]
                and self.conf["flush_threshold"] != -1
            ):
                self.flush()

            """save checkpoint file"""
            if self.conf["save_checkpoint"]:
                self.save_checkpoint()

        if len(self.valid_smiles_list) > 0:
            self.flush()

        if self.synonymous_threshold_num >0:
            if self.synonymous_counter > self.synonymous_threshold_num:
                '''test synonymous_trigger'''
                self.synonymous_trigger()
                self.synonymous_counter=0

    def insert_tokens_to_tree(self, single_smiles_token_list,reward):
        """
        从 root_node 开始，将 token_list 表示的路径插入到树中。
        参数：
            token_list: 例如 ["&", "C", "C", "O", "\n"]
            StateClass: 自定义的 State 类，用于构造新节点的状态
        返回：
            最终插入或命中的叶节点（Node）
        """
        print("add to tree")
        if single_smiles_token_list[-1] != "\n":
            single_smiles_token_list.append("\n")
        current_node = self.rootnode  # ✅ 初始化根节点，只执行一次
        for i, node_token in enumerate(single_smiles_token_list):
            if node_token == "&":
                continue

            matched_child = None
            for child in current_node.state.child_nodes:
                if child.position == node_token:
                    matched_child = child
                    break

            if matched_child is not None:
                current_node = matched_child
            else:
                new_state = State()
                new_state.position = current_node.state.position[:]
                new_state.add_position(node_token)
                new_state.parent_node = current_node
                current_node.add_node(node_token, new_state)
                current_node = current_node.state.child_nodes[-1]
        print("background")
        back_propagation(current_node, reward)
        print("add to tree finish")

    def synonymous_trigger(self):
        # === 用户配置 ===
        # 输入文件名
        print("strigeer start")
        # === 读取数据 ===
        df=pd.read_csv(self.output_path)
        select_num= self.conf.get("select_num", 10)
        synonymous_transform_times= self.conf.get("synonymous_transform_times", 5)


        # === 1. 按 reward 降序排序 ===
        df_sorted = df.sort_values(by="reward", ascending=False)

        # === 2. 取前 200 行 ===
        df_top_200 = df_sorted.head(select_num*4)


        # === 3. 去除重复 SMILES，只保留第一个 ===
        df_top_200["smiles"] = df_top_200["smiles"].apply(standardize_smiles)
        df_unique = df_top_200.drop_duplicates(subset="smiles", keep="first")

        # === 4. 取前 30 个 ===
        df_top_30 = df_unique.head(select_num)
        smiles_list=df_top_30["smiles"]
        reward_list=df_top_30["reward"]
        sme = SmilesEnumerator()
        for wait_smiles,reward in zip(smiles_list,reward_list):
                synonymous_mol_list=[]
                for i in range(synonymous_transform_times):
                    synonymous_mol = sme.randomize_smiles(wait_smiles)
                    synonymous_mol_list.append(synonymous_mol)
                unique_lst = list(set(synonymous_mol_list))
                for i in unique_lst:
                  tokenss=smi_tokenizer(i)
                  self.insert_tokens_to_tree(tokenss,reward)

                with open(self.output_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([wait_smiles, reward,synonymous_mol_list])
                print("strigger one smiles")
        print("trigger finish")




    def load_checkpoint(self):
        ckpt_path = os.path.join(self.conf["output_dir"], self.conf["checkpoint_file"])
        with open(ckpt_path, mode="rb") as f:
            cp_obj = pickle.load(f)
        self.gid = cp_obj["gid"]
        self.loop_counter_for_selection = cp_obj["loop_counter_for_selection"]
        self.loop_counter_for_expansion = cp_obj["loop_counter_for_expansion"]
        self.expanded_before = cp_obj["expanded_before"]
        self.start_time = cp_obj["start_time"]
        self.rootnode = cp_obj["rootnode"]
        self.conf = cp_obj["conf"]
        self.generated_dict = cp_obj["generated_dict"]
        self.total_valid_num = cp_obj["total_valid_num"]

    def save_checkpoint(self):
        ckpt_fname = self.conf["checkpoint_file"]
        ckpt_path = os.path.join(self.conf["output_dir"], ckpt_fname)
        stem, ext = ckpt_fname.rsplit(".", 1)
        # To keep the three most recent checkpoint files.
        ckpt1_path = os.path.join(self.conf["output_dir"], f"{stem}2.{ext}")
        ckpt2_path = os.path.join(self.conf["output_dir"], f"{stem}3.{ext}")
        if os.path.exists(ckpt1_path):
            os.rename(ckpt1_path, ckpt2_path)
        if os.path.exists(ckpt_path):
            os.rename(ckpt_path, ckpt1_path)

        cp_obj = {
            "gid": self.gid,
            "loop_counter_for_selection": self.loop_counter_for_selection,
            "loop_counter_for_expansion": self.loop_counter_for_expansion,
            "expanded_before": self.expanded_before,
            "start_time": self.start_time,
            "conf": self.conf,
            "rootnode": self.rootnode,
            "generated_dict": self.generated_dict,
            "total_valid_num": self.total_valid_num,
        }

        with open(ckpt_path, mode="wb") as f:
            pickle.dump(cp_obj, f)
        self.flush()

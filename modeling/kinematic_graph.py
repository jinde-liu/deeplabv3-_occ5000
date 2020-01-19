import torch
import torch.nn as nn
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
#######################################
# main procedure
#######################################
"""
1. user defines a human SKELETON graph
2. convert the SKELETON graph into a 'kinematic_graph' with the format like {[[parent_ids], part_id, 'part_name'], ...}
3. build modules for every human part in the 'kinematic_graph'
"""

# Skeleton tree, each entry in a list corresponds to the parts at the same level in the tree
# (parrent ID, part ID, part name)
# This skeleton should be ascending order
SKELETON = [
    [(-1, 0, 'torso')],
    [(0, 1, 'face')],
    [(1, 2, 'hair')],
    [(0, 3, 'left_arm'), (0, 4, 'right_arm')],
    [(3, 5, 'left_hand'), (4, 6, 'right_hand')],
    [(0, 7, 'left_leg'), (0, 8, 'right_leg')],
    [(7, 9, 'left_feet'), (8, 10, 'right_feet')],
    [(-1, 11, 'accessory')]
]

class _Block_with_one_input(nn.Module):
    """Base block class takes one input which comes from the backbone"""
    def __init__(self, input_dim, BatchNorm):
        super(_Block_with_one_input, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(input_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
        # self.conv_block2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.1))
        self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1)

        self._init_weight()

    def forward(self, x):
        # x = self.conv_block1(x)
        # x1 = self.conv_block2(x)
        # x = self.conv3(x1)
        x1 = self.conv_block1(x)
        x = self.conv3(x1)
        return x, x1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

class _Block_with_two_input(nn.Module):
    """Base block class takes two input, one from the backbone and one from the catenated parents input"""
    def __init__(self, input_dim, BatchNorm):
        super(_Block_with_two_input, self).__init__()
        self.conv_block1 = nn.Sequential(nn.Conv2d(input_dim, 256, kernel_size=3, stride=1, padding=1, bias=False),
                                       BatchNorm(256),
                                       nn.ReLU(),
                                       nn.Dropout(0.5))
        # self.conv_block2 = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1, bias=False),
        #                                BatchNorm(256),
        #                                nn.ReLU(),
        #                                nn.Dropout(0.1))
        self.conv3 = nn.Conv2d(256, 1, kernel_size=1, stride=1)
        self._init_weight()

    def forward(self, x, y):
        x = torch.cat((x, y), dim=1)
        # x = self.conv_block1(x)
        # x1 = self.conv_block2(x)
        # x = self.conv3(x1)
        x1 = self.conv_block1(x)
        x = self.conv3(x1)
        return x, x1

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()


#class SPL(nn.Module):
class Kinematic_graph(nn.Module):
    def __init__(self, BatchNorm):

        super(Kinematic_graph, self).__init__()
        # user input skeleton graph
        self.skeleton = SKELETON
        # kinematic graph, {[[parent_ids], part_id, 'part_name'], ...}
        self.kinematic_graph = dict()
        self._get_kinematic_graph()
        # one module for each human part in kinematic graph
        self.skeleton_blocks = dict()

        # build modules for every human part in the 'kinematic graph'
        self.torso = _Block_with_one_input(input_dim=304, BatchNorm=BatchNorm)
        self.face = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.hair = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.left_arm = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.right_arm = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.left_hand = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.right_hand = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.left_leg = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.right_leg = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.left_feet = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.right_feet = _Block_with_two_input(input_dim=560, BatchNorm=BatchNorm)
        self.accessory = _Block_with_one_input(input_dim=304, BatchNorm=BatchNorm)
        self.backgroud = _Block_with_one_input(input_dim=304, BatchNorm=BatchNorm)
        """
        for i in range(len(self.kinematic_graph)):
            if self.kinematic_graph[i][0]:
                temp_block = _Block_with_two_input(input_dim=512, BatchNorm=BatchNorm)
                self.skeleton_blocks[self.kinematic_graph[i][1]] = temp_block
            else:
                temp_block = _Block_with_one_input(input_dim=256, BatchNorm=BatchNorm)
                self.skeleton_blocks[self.kinematic_graph[i][1]] = temp_block
        """
        self._init_weight()

    def forward(self, x):
        """Build the forward pass of the kinematic graph by hand, maybe modified to an easier way in the future"""

        torso, torso_mid_level = self.torso(x)
        face, face_mid_level = self.face(x, torso_mid_level)
        hair, hair_mid_level = self.hair(x, face_mid_level)
        left_arm, left_arm_mid_level = self.left_arm(x, torso_mid_level)
        right_arm, right_arm_mid_level = self.right_arm(x, torso_mid_level)
        left_hand, left_hand_mid_level = self.left_hand(x, left_arm_mid_level)
        right_hand, right_hand_mid_level = self.right_hand(x, right_arm_mid_level)
        left_leg, left_leg_mid_level = self.left_leg(x, torso_mid_level)
        right_leg, right_leg_mid_level = self.right_leg(x, torso_mid_level)
        left_feet, left_feet_mid_level = self.left_feet(x, left_leg_mid_level)
        right_feet, right_feet_mid_level = self.right_feet(x, right_leg_mid_level)
        accessory, _ = self.accessory(x)
        background, _ = self.backgroud(x)

        """ Class IDs in occ5000 dataset, not same as the IDs in SKELETON
        background: 0
        hair: 1
        face: 2
        torso: 3
        left_arm: 4
        right_arm: 5
        left_hand: 6
        right_hand: 7
        left_leg: 8
        right_leg: 9
        left_foot: 10
        right_foot: 11
        accessory: 12
        """
        return torch.cat((background, hair, face, torso, left_arm, right_arm, left_hand, right_hand,
                          left_leg, right_leg, left_feet, right_feet, accessory), dim=1)

    def _get_kinematic_graph(self):
        """Get kinematic graph, {[[parent_ids], part_id, 'part_name'], ...}
        """
        indexed_skeleton = dict()
        for human_parts in self.skeleton:
            for human_part in human_parts:
                parent_list_ = [human_part[0]] if human_part[0] > -1 else []
                # This line makes sure the human par id is same as the list id
                indexed_skeleton[human_part[1]] = [parent_list_, human_part[1], human_part[2]]

        def get_all_parents(parent_list, parent_id, tree):
            if parent_id not in parent_list:
                parent_list.append(parent_id)
                for parent in tree[parent_id][0]:
                    get_all_parents(parent_list, parent, tree)

        # Get kinematic graph with all parent parts, {[[parent_ids], part_id, 'part_name'], ...}
        for i in range(len(indexed_skeleton)):
            human_part = indexed_skeleton[i]
            parent_list = list()
            if len(human_part[0]) > 0:
                get_all_parents(parent_list, human_part[0][0], indexed_skeleton)
            new_human_part = [parent_list, human_part[1], human_part[2]]
            self.kinematic_graph[i] = new_human_part

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                # m.weight.data.normal_(0, math.sqrt(2. / n))
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, SynchronizedBatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

def build_kinematic_graph(BatchNorm):
    return Kinematic_graph(BatchNorm)
if __name__ == '__main__':
    from tensorboardX import SummaryWriter
    batchnorm = nn.BatchNorm2d
    writer = SummaryWriter('/home/kidd/Documents/graph')
    kp = build_kinematic_graph(batchnorm)
    input = torch.randn(1,304, 33, 33)
    writer.add_graph(kp, input)
    output = kp(input)
    print(output.shape)

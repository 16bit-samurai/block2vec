#!/usr/bin/env python
from torch import nn
from torch.utils.data import IterableDataset, DataLoader
import collections
from mcworldreader import World
import itertools
import os
import mysql.connector
import time
import atexit
import random
import math
import logging
import modelscope
from mcworldreader import World
from multiprocessing.pool import ThreadPool


console_handler = logging.StreamHandler()
file_handler = logging.FileHandler('import.log', mode='w+')

logging.basicConfig(level=logging.INFO, handlers=[console_handler, file_handler], format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

connection = mysql.connector.connect(
    host="127.0.0.1",
    port=3188,
    user=os.environ["B2V_USER"],
    password=os.environ["B2V_PASSWD"],
    database="defaults",
    auth_plugin="mysql_native_password"
)

cursor = connection.cursor()

class BlockDropOut():
    def __init__(self):
        self.total = 0
        self.count = [0] * 1162

    def should_drop(self, block_idx):
        self.total += 1
        self.count[block_idx] += 1
        frq = self.count[block_idx] / self.total
        drop_pec = 1 - (0.00001 / frq) ** 1 / 2
        if random.random() < drop_pec:
            return True

        return False


def is_skip_block(block):
    if block == "minecraft:air":
        return True

    return False


def tokenizer():
    vocab = collections.OrderedDict()
    with open('vocab.txt', mode='r') as f:
        tokens = f.readlines()
    
    for index, token in enumerate(tokens):
        token = token.rstrip()
        vocab[token] = index

    return vocab


def get_nearby_region(world, coord):
    regions = []
    coord_list = itertools.product([-1, 0, 1], [-1, 0, 1])

    for c in coord_list:
        region = world.get_region(coord[0] + c[0], coord[1] + c[1])
        if not region is None:
            regions.append(region)

    return regions


def get_nearby_block(world, win_size, x, y, z):
    blocks = []

    if win_size == 1:
        for c in [(1, 0, 0), (0, 1, 0), (0, 0, 1), (-1, 0, 0), (0, -1, 0), (0, 0, -1)]:
            block = world.get_block(x + c[0], y + c[1], z + c[2])
            if not block is None:
                blocks.append(block)

    elif win_size == 2:
        for c in itertools.product([-1, 0, 1], [-1, 0, 1], [-1, 0, 1]):
            if c == (0, 0, 0):
                continue

            block = world.get_block(x + c[0], y + c[1], z + c[2])
            if not block is None:
                blocks.append(block)
        
        for c in ([2, 0, 0], [0, 2, 0], [0, 0, 2], [-2, 0, 0], [0, 0, -2]):
            block = world.get_block(x + c[0], y + c[1], z + c[2])
            if not block is None:
                blocks.append(block)

    return blocks


def parse_region(tup):
    world, coord, region = tup
    for data, label in iter_region(world, coord, region):
        cursor.execute('INSERT INTO blocks(center, target) VALUES ({}, {});'.format(data, label))

    logger.info("{} {}".format(coord[0], coord[1]))
    connection.commit()


def iter_region(world, coord, region):
    nearbys = get_nearby_region(world, coord)
    for (x, y, z), block in region.iter_all_blocks():
        center = tokenizer[block]
        if is_skip_block(block):
            continue

        for block in get_nearby_block(world, 2, x, y, z):
            target = tokenizer[block]
            if target == center:
                continue
            
            if dropout.should_drop(center):
                continue
            
            yield center, target
            if is_skip_block(block):
                yield target, center

    region.empty_cache()
    for n in nearbys:
        n.empty_cache()


def close_connection(connection):
    connection.close()


if __name__ == "__main__":
    world = World(".world_big")
    tokenizer = tokenizer()
    dropout = BlockDropOut()

    atexit.register(close_connection, connection)

    region_lists = world.get_region_list()
    for coord, region in region_lists:
        parse_region((world, coord, region))

    
    # with ThreadPool(8) as p:
    #     p.map(parse_region, region_lists)


#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
Detectron2 training script with a plain training loop.

This script reads a given config file and runs the training or evaluation.
It is an entry point that is able to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as a library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.

Compared to "train_net.py", this script supports fewer default features.
It also includes fewer abstraction, therefore is easier to add custom logic.
"""

import logging
import os
from collections import OrderedDict
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.data.common import DatasetFromList
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

from network import MyNetwork

logger = logging.getLogger("detectron2")


# seems it has to be a nn.Module to use torch optimizers
class RoutingWeightModel(nn.Module):
    def __init__(self, a, b):
        super(RoutingWeightModel, self).__init__()
        self.routing_weights = nn.Parameter(torch.zeros(a, b))


def get_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        assert (
            torch.cuda.device_count() >= comm.get_rank()
        ), "CityscapesEvaluator currently do not work with multiple machines."
        return CityscapesSemSegEvaluator(dataset_name)
    if evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    if evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, cfg, True, output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    if len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)


def do_test(cfg, model):
    results = OrderedDict()
    for dataset_name in cfg.DATASETS.TEST:
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    model_weights = torch.load(cfg.MODEL.WEIGHTS)
    if "model" in model_weights:
        model_weights = model_weights["model"]
    model.load_state_dict(model_weights, strict=False)  # should better set True for once to see if it's loaded right

    assert cfg.SOLVER.IMS_PER_BATCH == 1, f"should set batchsize=1"
    sampler = torch.utils.data.sampler.SequentialSampler(1725)
    data_loader = build_detection_train_loader(cfg, sampler=sampler, aspect_ratio_grouping=False)
    num_images = len(data_loader)

    routing_weights_model = RoutingWeightModel(num_images, 24)
    routing_weights_model = routing_weights_model.to(torch.device(cfg.MODEL.DEVICE))
    optimizer = torch.optim.SGD(routing_weights_model.parameters(), lr=cfg.SOLVER.BASE_LR,
                                momentum=cfg.SOLVER.MOMENTUM, weight_decay=cfg.SOLVER.WEIGHT_DECAY)

    logger.info("Starting solving optimized routing weights")

    for data, iteration in zip(data_loader, range(num_images)):
        for _ in range(10):
            w = routing_weights_model.routing_weights[iteration]
            w_list = [
                torch.sigmoid(w[:8])[None, :],
                torch.sigmoid(w[8:16])[None, :],
                torch.sigmoid(w[16:])[None, :],
            ]
            data[0]["routing_weights"] = w_list
            loss_dict = model(data)
            losses = sum(loss_dict.values())
            print(type(loss_dict), loss_dict, type(losses), losses)
            assert torch.isfinite(losses).all(), loss_dict

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            print(iteration, losses.item())
    
    routing_weights = routing_weights_model.routing_weights+0
    torch.save(routing_weights, "optimal_routing_weights.pth")
    return routing_weights


def setup(args):
    """
    Create configs and perform basic setups.
    """
    register_coco_instances("domain", {}, "domain/annotations.json", "domain")
    register_coco_instances("domain_train", {}, "domain/train_annotations.json", "domain")
    register_coco_instances("domain_test", {}, "domain/test_annotations.json", "domain")
    register_coco_instances("routine_train", {}, "domain/train_routine_5fc766.json", "domain")
    register_coco_instances("routine_test", {}, "domain/test_routine_5fc877.json", "domain")
    cfg = get_cfg()
    assert args.config_file == "", f"This code automatically uses the config file in this directory"
    args.config_file = "configs.yaml"
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(
        cfg, args
    )  # if you don't like any of the default setup, write your own setup code
    return cfg


def main(args):
    cfg = setup(args)

    model = MyNetwork(cfg)
    model.to(torch.device(cfg.MODEL.DEVICE))
    logger.info("Model:\n{}".format(model))
    if args.eval_only:
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        return do_test(cfg, model)

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    return do_train(cfg, model, resume=args.resume)


if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )

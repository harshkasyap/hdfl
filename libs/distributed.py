import copy
import os
import sys
import time
import socket
from builtins import len, range, super

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
from libs import nn
from libs.protobuf_producer import *
from libs.protobuf_consumer import *

#!pip install syft
import syft as sy
from syft.federated.model_serialization import State
from syft.federated.model_serialization import wrap_model_params

class Distributed:
    def __init__(self, clients, brokers_ip, schema_ip, wait_to_consume):
        self.broker_ip = brokers_ip
        self.schema_ip = schema_ip
        self.group_prefix = "group-"
        self.client_prefix = "client-"        
        self.clients = clients[:]
        self.wait_to_consume = wait_to_consume

    def is_node_avail(self, node_index):
        if node_index in self.clients:
            return True
        print("Client {} is not available for training".format(node_index))
        return False
            
    def consume_model(self, node_index, _topic, _model, _epoch):
        _rcvd_models = {}

        if self.is_node_avail(node_index):
            group_id = self.group_prefix + node_index
            client_id = self.client_prefix + node_index
            protobuf_consumer = ProtobufConsumer(group_id, client_id, State.get_protobuf_schema(), _topic, self.broker_ip)

            try:
                t_end = time.time() + self.wait_to_consume
                while time.time() < t_end:
                    msg = protobuf_consumer.consume()
                    epoch = -1
                    if msg is not None:
                        if msg.headers() is not None:
                            for tup in msg.headers():
                                if tup[0] == 'iteration' and tup[1] is not None:
                                    epoch = tup[1].decode('utf-8')

                    if epoch != -1 or str(epoch) == str(_epoch):
                        model = nn.getModel(msg.value(), _model)
                        _rcvd_models[msg.key()] = copy.deepcopy(model)
            except KeyboardInterrupt:
                print("Exception KeyboardInterrupt occured consuming update for client {}".format(node_index))
            except Exception as se:
                print("Exception {} occured consuming update for client {}".format(se, node_index))
            finally:
                protobuf_consumer.close()
        
        return _rcvd_models
    
    def produce_model(self, node_index, _topic, model, epoch):
        if self.is_node_avail(node_index):
            group_id = self.group_prefix + node_index
            client_id = self.client_prefix + node_index
            pb = sy.serialize(wrap_model_params(model.parameters()))
            protobuf_producer = ProtobufProducer(group_id, client_id, State.get_protobuf_schema(), self.broker_ip, self.schema_ip)
            try:
                protobuf_producer.produce(topic=_topic, 
                                          _key=str(node_index), 
                                          _value=pb, 
                                          _headers={'iteration': str(epoch)})
            except Exception as se:
                print("Exception {} occured producing update for client {}".format(se, node_index))
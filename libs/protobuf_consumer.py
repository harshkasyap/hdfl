#!pip install confluent-kafka
from confluent_kafka import KafkaError
from confluent_kafka import DeserializingConsumer
from confluent_kafka.serialization import StringDeserializer
from confluent_kafka.schema_registry.protobuf import ProtobufDeserializer

#!pip install syft
#import syft as sy

class ProtobufConsumer():

    def __init__(self, group_id, client_id, _protobuf_schema, topic,
                 booststraps_servers = '172.16.26.40:9092'):

        self.protobuf_schema = _protobuf_schema
        self.protobuf_deserializer = ProtobufDeserializer(self.protobuf_schema)

        self._key_deserializer = StringDeserializer('utf_8')
        self._value_deserializer = self.protobuf_deserializer
        
        self.consumer_conf = {'bootstrap.servers': booststraps_servers,
                              'key.deserializer': self._key_deserializer,
                              'value.deserializer': self._value_deserializer,
                              'group.id': group_id,
                              'client.id': client_id,
                              'enable.auto.commit': True,
                              'session.timeout.ms': 6000,
                              'default.topic.config': {'auto.offset.reset': 'earliest'}}
        
        self.protobuf_consumer = DeserializingConsumer(self.consumer_conf)
        self.protobuf_consumer.subscribe([topic])

    def consume(self):
        msg = self.protobuf_consumer.poll(0.1)
        if msg is None:
            return None
        elif not msg.error():
            return msg
        elif msg.error().code() == KafkaError._PARTITION_EOF:
            log.error('End of partition reached {0}/{1}'
                  .format(msg.topic(), msg.partition()))
            return None
        else:
            log.error('Error occured: {0}'.format(msg.error().str()))
            return None
            
    def close(self):
        self.protobuf_consumer.close()
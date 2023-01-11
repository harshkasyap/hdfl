#!pip install confluent-kafka
from confluent_kafka import SerializingProducer
from confluent_kafka.serialization import StringSerializer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.protobuf import ProtobufSerializer

from libs import log

class ProtobufProducer():

    def __init__(self, group_id, client_id, _protobuf_schema,
                 booststraps_servers = '172.16.26.40:9092',
                 _schema_registry_conf = 'http://172.16.26.40:8081'):

        self.schema_registry_conf = {'url': _schema_registry_conf}
        self.schema_registry_client = SchemaRegistryClient(self.schema_registry_conf)
        self.protobuf_schema = _protobuf_schema
        self.protobuf_serializer = ProtobufSerializer(self.protobuf_schema, self.schema_registry_client)
        
        self._key_serializer = StringSerializer('utf_8')
        self._value_serializer = self.protobuf_serializer
        
        self.producer_conf = {'bootstrap.servers': booststraps_servers,
                              'key.serializer': self._key_serializer,
                              'value.serializer': self._value_serializer,
                              'compression.codec' : 'snappy',
                              'message.max.bytes': 1024*1024*10,
                              'group.id': group_id,
                              'client.id': client_id}
        
        self.protobuf_producer = SerializingProducer(self.producer_conf)
        
    def delivery_report(err, msg):
        """
        Reports the failure or success of a message delivery.
        Args:
            err (KafkaError): The error that occurred on None on success.
            msg (Message): The message that was produced or failed.
        Note:
            In the delivery report callback the Message.key() and Message.value()
            will be the binary format as encoded by any configured Serializers and
            not the same object that was passed to produce().
            If you wish to pass the original object(s) for key and value to delivery
            report callback we recommend a bound callback or lambda where you pass
            the objects along.
        """
        if err is not None:
            log.error("Delivery failed for User record {}: {}".format(msg.key(), err))
            return
        log.info('User record {} successfully produced to {} [{}] at offset {}'.format(
            msg.key(), msg.topic(), msg.partition(), msg.offset()))

    def produce(self, topic, _key=None, _value=None, partition=-1,
                _on_delivery=delivery_report, timestamp=0, _headers=None):

        log.info("Producing user records to topic {}. ^C to exit.".format(topic))

        try:
            self.protobuf_producer.produce(topic=topic, key=_key, value=_value, 
                                           on_delivery=_on_delivery, headers =_headers)
        except ValueError:
            log.error("Invalid input, discarding record...")
        except Exception as se:
            log.error("Exception raised")

        log.info("Flushing records...")
        self.protobuf_producer.flush()
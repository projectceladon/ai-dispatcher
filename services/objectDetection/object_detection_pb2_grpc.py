# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
import grpc

import object_detection_pb2 as object__detection__pb2


class DetectionStub(object):
  """The Detection service definition.
  """

  def __init__(self, channel):
    """Constructor.

    Args:
      channel: A grpc.Channel.
    """
    self.getPredictions = channel.unary_unary(
        '/objectDetection.Detection/getPredictions',
        request_serializer=object__detection__pb2.RequestBytes.SerializeToString,
        response_deserializer=object__detection__pb2.PredictionsList.FromString,
        )


class DetectionServicer(object):
  """The Detection service definition.
  """

  def getPredictions(self, request, context):
    """
    """
    context.set_code(grpc.StatusCode.UNIMPLEMENTED)
    context.set_details('Method not implemented!')
    raise NotImplementedError('Method not implemented!')


def add_DetectionServicer_to_server(servicer, server):
  rpc_method_handlers = {
      'getPredictions': grpc.unary_unary_rpc_method_handler(
          servicer.getPredictions,
          request_deserializer=object__detection__pb2.RequestBytes.FromString,
          response_serializer=object__detection__pb2.PredictionsList.SerializeToString,
      ),
  }
  generic_handler = grpc.method_handlers_generic_handler(
      'objectDetection.Detection', rpc_method_handlers)
  server.add_generic_rpc_handlers((generic_handler,))

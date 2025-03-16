import 'package:flutter/material.dart';
import 'package:pedometer/pedometer.dart';
import 'package:geolocator/geolocator.dart';
import 'package:health/health.dart';

void main() {
  runApp(const HealthApp());
}

class HealthApp extends StatelessWidget {
  const HealthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.blue, useMaterial3: true),
      home: const HealthScreen(),
    );
  }
}

class HealthScreen extends StatefulWidget {
  const HealthScreen({super.key});

  @override
  State<HealthScreen> createState() => _HealthScreenState();
}

class _HealthScreenState extends State<HealthScreen> {
  int _stepCount = 0;
  double? _speed;
  double? _latitude;
  double? _longitude;
  int? _heartRate;

  @override
  void initState() {
    super.initState();
    _getStepCount();
    _getLocation();
    _getHeartRate();
  }

  void _getStepCount() {
    Pedometer.stepCountStream.listen((StepCount stepCount) {
      setState(() {
        _stepCount = stepCount.steps;
      });
    });
  }

  Future<void> _getLocation() async {
    bool serviceEnabled = await Geolocator.isLocationServiceEnabled();
    if (!serviceEnabled) return;

    LocationPermission permission = await Geolocator.checkPermission();
    if (permission == LocationPermission.denied) {
      permission = await Geolocator.requestPermission();
      if (permission == LocationPermission.deniedForever) return;
    }

    Geolocator.getPositionStream().listen((Position position) {
      setState(() {
        _latitude = position.latitude;
        _longitude = position.longitude;
        _speed = position.speed;
      });
    });
  }

  Future<void> _getHeartRate() async {
    Health health = Health();
    List<HealthDataType> types = [HealthDataType.HEART_RATE];

    bool authorized = await health.requestAuthorization(types);
    if (authorized) {
      List<HealthDataPoint> healthData = await health.getHealthDataFromTypes(
      types: types,
      startTime: DateTime.now().subtract(const Duration(minutes: 5)),
      endTime: DateTime.now(),
      );

      if (healthData.isNotEmpty) {
        setState(() {
          _heartRate = (healthData.last.value as double).round();
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Health Tracker", style: TextStyle(fontWeight: FontWeight.bold))),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            _buildCard("Passi", "$_stepCount", Icons.directions_walk, Colors.green),
            _buildCard("Velocità", "${_speed?.toStringAsFixed(2) ?? "N/A"} m/s", Icons.speed, Colors.blue),
            _buildCard("Latitudine", "${_latitude?.toStringAsFixed(5) ?? "N/A"}", Icons.map, Colors.orange),
            _buildCard("Longitudine", "${_longitude?.toStringAsFixed(5) ?? "N/A"}", Icons.map_outlined, Colors.purple),
            _buildCard("Frequenza Cardiaca", "${_heartRate ?? "N/A"} bpm", Icons.favorite, Colors.red),
          ],
        ),
      ),
    );
  }

  Widget _buildCard(String title, String value, IconData icon, Color color) {
    return Card(
      elevation: 4,
      shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(15)),
      child: ListTile(
        leading: Icon(icon, color: color, size: 40),
        title: Text(title, style: const TextStyle(fontWeight: FontWeight.bold, fontSize: 18)),
        subtitle: Text(value, style: const TextStyle(fontSize: 16)),
      ),
    );
  }
}

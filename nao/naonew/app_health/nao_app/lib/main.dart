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
      home: const LoginPage(),
    );
  }
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();

  // Credenziali predefinite
  final String _username = "Mattia";
  final String _password = "12345";

  void _login() {
    // Verifica le credenziali
    if (_usernameController.text == _username && _passwordController.text == _password) {
      // Naviga alla schermata principale
      Navigator.pushReplacement(
        context,
        MaterialPageRoute(builder: (context) => const HealthScreen()),
      );
    } else {
      // Mostra errore se le credenziali sono sbagliate
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Credenziali errate!')),
      );
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text("Login"),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            TextField(
              controller: _usernameController,
              decoration: const InputDecoration(
                labelText: 'Nome utente',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.person),
              ),
            ),
            const SizedBox(height: 20),
            TextField(
              controller: _passwordController,
              obscureText: true,
              decoration: const InputDecoration(
                labelText: 'Password',
                border: OutlineInputBorder(),
                prefixIcon: Icon(Icons.lock),
              ),
            ),
            const SizedBox(height: 30),
            ElevatedButton(
              onPressed: _login,
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(horizontal: 50, vertical: 15),
              ),
              child: const Text('Accedi'),
            ),
          ],
        ),
      ),
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
    Pedometer.stepCountStream.listen((stepCount) {
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

    // Definiamo il tipo di dati da richiedere
    List<HealthDataType> types = [HealthDataType.HEART_RATE];

    // Richiediamo l'autorizzazione per accedere ai dati
    bool authorized = await health.requestAuthorization(types);

    if (authorized) {
      DateTime now = DateTime.now();
      DateTime startTime = now.subtract(const Duration(minutes: 10));

      // Ottieni i dati relativi alla frequenza cardiaca
      List<HealthDataPoint> healthData = await health.getHealthDataFromTypes(
        types: types,
        startTime: startTime,
        endTime: now,
      );

      if (healthData.isNotEmpty) {
        setState(() {
          _heartRate = (healthData.last.value as double).round();
        });
      }
    } else {
      print("Permessi negati per l'accesso ai dati sanitari");
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

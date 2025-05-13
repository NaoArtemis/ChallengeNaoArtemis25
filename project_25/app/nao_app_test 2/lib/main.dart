import 'package:flutter/material.dart';
import 'package:google_sign_in/google_sign_in.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

void main() {
  runApp(const HealthApp());
}

class HealthApp extends StatelessWidget {
  const HealthApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      theme: ThemeData(primarySwatch: Colors.blue),
      home: const LoginPage(),
    );
  }
}

class User {
  final String id;
  final String username;
  final String password;
  int? heartRate;
  int? steps;
  double? speed;

  User({
    required this.id,
    required this.username,
    required this.password,
    this.heartRate,
    this.steps,
    this.speed,
  });

  Map<String, dynamic> toJson() => {
        'heartRate': heartRate,
        'steps': steps,
        'speed': speed,
      };
}

class LoginPage extends StatefulWidget {
  const LoginPage({super.key});

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final _usernameController = TextEditingController();
  final _passwordController = TextEditingController();

  void _login() async {
    final username = _usernameController.text;
    final password = _passwordController.text;

    if (username.isEmpty || password.isEmpty) return;

    final url = Uri.parse('http://192.168.0.103:5010/api/app/utente/$username');

    try {
      final response = await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode({'username': username, 'password': password}),
      );

      if (response.statusCode == 200) {
        final data = json.decode(response.body);
        if (data['code'] == 200) {
          final user = User(
            id: data['data']['id_player'].toString(),
            username: username,
            password: password,
          );

          Navigator.pushReplacement(
            context,
            MaterialPageRoute(
                builder: (context) => HealthScreen(user: user)),
          );
        }
      }
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: Center(
        child: Padding(
          padding: const EdgeInsets.all(16),
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextField(
                controller: _usernameController,
                decoration: const InputDecoration(labelText: 'Username'),
              ),
              TextField(
                controller: _passwordController,
                obscureText: true,
                decoration: const InputDecoration(labelText: 'Password'),
              ),
              const SizedBox(height: 20),
              ElevatedButton(
                onPressed: _login,
                child: const Text('Login'),
              ),
            ],
          ),
        ),
      ),
    );
  }
}

class HealthScreen extends StatefulWidget {
  final User user;
  const HealthScreen({super.key, required this.user});

  @override
  State<HealthScreen> createState() => _HealthScreenState();
}

class _HealthScreenState extends State<HealthScreen> {
  late User _user;
  late Timer _timer;

  @override
  void initState() {
    super.initState();
    _user = widget.user;
    _fetchGoogleFitData();
    _timer = Timer.periodic(const Duration(minutes: 1), (_) => _sendData());
  }

  @override
  void dispose() {
    _timer.cancel();
    super.dispose();
  }

  Future<void> _fetchGoogleFitData() async {
    try {
      final googleSignIn = GoogleSignIn(
        scopes: [
          'https://www.googleapis.com/auth/fitness.heart_rate.read',
          'https://www.googleapis.com/auth/fitness.activity.read',
          'email',
        ],
      );

      final account = await googleSignIn.signIn();
      if (account == null) return;
      final auth = await account.authentication;

      final now = DateTime.now();
      final start = now.subtract(const Duration(hours: 24));
      final startNs = (start.millisecondsSinceEpoch * 1000000).toString();
      final endNs = (now.millisecondsSinceEpoch * 1000000).toString();

      // Heart Rate
      final heartUrl = Uri.parse(
          'https://www.googleapis.com/fitness/v1/users/me/dataSources/derived:com.google.heart_rate.bpm:com.google.android.gms:merge_heart_rate_bpm/datasets/$startNs-$endNs');
      final heartResponse = await http.get(heartUrl,
          headers: {'Authorization': 'Bearer ${auth.accessToken}'});
      if (heartResponse.statusCode == 200) {
        final points = json.decode(heartResponse.body)['point'];
        if (points != null && points.isNotEmpty) {
          final bpm = points.last['value'][0]['fpVal'];
          _user.heartRate = (bpm is num) ? bpm.round() : 0;
        }
      }

      // Steps
      final stepsUrl = Uri.parse(
          'https://www.googleapis.com/fitness/v1/users/me/dataSources/derived:com.google.step_count.delta:com.google.android.gms:estimated_steps/datasets/$startNs-$endNs');
      final stepsResponse = await http.get(stepsUrl,
          headers: {'Authorization': 'Bearer ${auth.accessToken}'});
      if (stepsResponse.statusCode == 200) {
        final points = json.decode(stepsResponse.body)['point'];
        int totalSteps = 0;
        for (var point in points) {
          final stepValue = point['value']?[0]?['intVal'];
          if (stepValue != null && stepValue is int) {
            totalSteps += stepValue;
          }
        }
        _user.steps = totalSteps;
      }

      // Speed (mocked as not always available from Google Fit)
      _user.speed = 0.0;

      setState(() {});
    } catch (_) {}
  }

  Future<void> _sendData() async {
    await _fetchGoogleFitData();
    final url = Uri.parse('http://192.168.0.103:5010/api/app/dati/${_user.id}');

    try {
      await http.post(
        url,
        headers: {'Content-Type': 'application/json'},
        body: json.encode(_user.toJson()),
      );
    } catch (_) {}
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text("Health Monitor")),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            ElevatedButton(
              onPressed: _fetchGoogleFitData,
              child: const Text("Accedi a Google Fit"),
            ),
            const SizedBox(height: 20),
            Text("Heart Rate: ${_user.heartRate ?? 'N/A'} bpm"),
            Text("Steps: ${_user.steps ?? 'N/A'}"),
            Text("Speed: ${_user.speed ?? 'N/A'} m/s"),
          ],
        ),
      ),
    );
  }
}